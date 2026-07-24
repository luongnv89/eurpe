import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AlertCircle, CheckCircle2, Download, Loader2, RefreshCw, Sparkles } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { WorkspaceHeader } from "@/components/WorkspaceHeader";
import { cn } from "@/lib/utils";

import {
  fetchCallContext,
  fetchGenerationEnums,
  fetchProfiles,
  generateSection,
  iterateSection,
} from "./api";
import type {
  GenerateSectionRequest,
  GenerateSectionResponse,
  GenerationEnumsResponse,
  IterateSectionRequest,
  ProfilesResponse,
} from "./api";
import { DraftPreview } from "./DraftPreview";

// Sentinel value used by the Select to express "no programme / no profile
// chosen". Radix Select Items refuse an empty-string value, so we adopt
// the convention used by other shadcn templates and translate the
// sentinel back to ``null`` before posting to the server.
const NONE_VALUE = "__none__";

// Listed up front so the AC #1 invariant ("UI supports section type,
// drafting profile, call/topic input, and user intent fields") is one
// diff line away from review.
const REQUIRED_FIELDS = ["section_type", "user_intent"] as const;

const DEFAULT_TOP_K = 5;

// AC #1 of issue #16: "User can set critic iterations between 1 and
// 5 before generation." Defaults to 3 per the issue body. Mirrors
// the server-side cap (Pydantic ge=1, le=5 on max_iterations).
const DEFAULT_MAX_ITERATIONS = 3;
const MIN_ITERATIONS = 1;
const MAX_ITERATIONS = 5;

type ContextMode = "free" | "structured";

interface StructuredContext {
  callId: string;
  topicId: string;
  topicTitle: string;
  expectedOutcomes: string;
  scope: string;
}

const EMPTY_STRUCTURED: StructuredContext = {
  callId: "",
  topicId: "",
  topicTitle: "",
  expectedOutcomes: "",
  scope: "",
};

/**
 * Compose the structured context fields into the single free-text
 * ``call_context`` string the backend route accepts.
 *
 * The structured tab is a UX nicety: it gives the operator a guided
 * form with labelled fields, but server-side everything funnels into
 * the same ``call_context`` blob the generation workflow consumes. We
 * format with explicit headings so the LLM has the same cues it would
 * see in the CLI ``--topic-text`` path (see
 * :mod:`eurpe.intake.extractor`).
 */
function buildStructuredContextString(value: StructuredContext): string {
  const sections: string[] = [];
  if (value.callId.trim()) sections.push(`Call: ${value.callId.trim()}`);
  if (value.topicId.trim()) sections.push(`Topic ID: ${value.topicId.trim()}`);
  if (value.topicTitle.trim()) sections.push(`Topic: ${value.topicTitle.trim()}`);
  if (value.expectedOutcomes.trim()) {
    sections.push(`Expected Outcomes:\n${value.expectedOutcomes.trim()}`);
  }
  if (value.scope.trim()) {
    sections.push(`Scope:\n${value.scope.trim()}`);
  }
  return sections.join("\n\n");
}

/**
 * Section-drafting workspace — the React side of Task 3.1.
 *
 * Layout:
 *
 * 1. ``Section type`` Select — drives the workflow's ``section_type``.
 * 2. ``Drafting profile`` Select — drives ``profile_programme``.
 * 3. ``Call / topic context`` Tabs — operator picks free-text vs a
 *    guided structured form. Both modes compose down to the route's
 *    ``call_context`` field server-side.
 * 4. ``User intent`` Textarea — the primary required field. Drives
 *    retrieval and is quoted verbatim in the prompt.
 * 5. ``Retrieval`` knobs — programme filter, top-k, lessons-learned.
 *
 * Validation runs client-side first (AC #4) and the route's Pydantic
 * model re-checks server-side as a safety net.
 */
export function DraftingWorkspace() {
  const [enums, setEnums] = useState<GenerationEnumsResponse | null>(null);
  const [enumsError, setEnumsError] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ProfilesResponse | null>(null);
  const [profilesError, setProfilesError] = useState<string | null>(null);

  const [sectionType, setSectionType] = useState<string>("");
  const [profileProgramme, setProfileProgramme] = useState<string>(NONE_VALUE);
  const [targetProgramme, setTargetProgramme] = useState<string>(NONE_VALUE);
  const [userIntent, setUserIntent] = useState<string>("");
  const [contextMode, setContextMode] = useState<ContextMode>("free");
  const [freeContext, setFreeContext] = useState<string>("");
  const [structured, setStructured] = useState<StructuredContext>(EMPTY_STRUCTURED);
  const [topK, setTopK] = useState<string>(String(DEFAULT_TOP_K));
  const [lessonsLearned, setLessonsLearned] = useState<boolean>(false);
  // Issue #16: configurable critic loop (AC #1).
  const [maxIterations, setMaxIterations] = useState<string>(
    String(DEFAULT_MAX_ITERATIONS),
  );

  const [clientErrors, setClientErrors] = useState<string[]>([]);
  const [generating, setGenerating] = useState<boolean>(false);
  const [refining, setRefining] = useState<boolean>(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [draft, setDraft] = useState<GenerateSectionResponse | null>(null);
  // Issue #16: AC #2 — operator accepts the current draft and the
  // loop stops. ``stoppedAtCap`` mirrors the server's ``stopped``
  // flag so the UI can show "iteration cap reached" alongside the
  // user's explicit accept.
  const [accepted, setAccepted] = useState<boolean>(false);
  const [stoppedAtCap, setStoppedAtCap] = useState<boolean>(false);

  // Issue #67: auto-fill structured-context fields from a pasted
  // Funding & Tenders Portal URL. ``fetchHint`` shows the post-success
  // "paste outcomes/scope manually" nudge because the portal does not
  // expose those two fields via its public API for current Horizon
  // Europe topics (see eurpe/intake/call_fetcher.py for the rationale).
  const [callUrl, setCallUrl] = useState<string>("");
  const [fetchingCall, setFetchingCall] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [fetchHint, setFetchHint] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchGenerationEnums(ctrl.signal)
      .then(setEnums)
      .catch((err) => {
        if (err?.name === "AbortError" || ctrl.signal.aborted) return;
        setEnumsError(err.message ?? String(err));
      });
    fetchProfiles(ctrl.signal)
      .then(setProfiles)
      .catch((err) => {
        if (err?.name === "AbortError" || ctrl.signal.aborted) return;
        setProfilesError(err.message ?? String(err));
      });
    return () => ctrl.abort();
  }, []);

  const profileItems = useMemo(() => profiles?.profiles ?? [], [profiles]);
  const sectionTypes = useMemo(() => enums?.section_type ?? [], [enums]);
  const programmes = useMemo(() => enums?.programme ?? [], [enums]);

  // Issue #67: paste a portal URL, fetch the call/topic triple, drop it
  // into the structured tab, switch the Tabs to ``structured`` so the
  // operator sees the result without an extra click.
  const handleFetchCall = useCallback(async () => {
    const trimmed = callUrl.trim();
    if (!trimmed) return;
    setFetchingCall(true);
    setFetchError(null);
    setFetchHint(null);
    try {
      const result = await fetchCallContext(trimmed);
      setStructured({
        callId: result.call_id,
        topicId: result.topic_id,
        topicTitle: result.topic_title,
        expectedOutcomes: result.expected_outcomes,
        scope: result.scope,
      });
      setContextMode("structured");
      // Always-empty fields in v1; tell the operator why so the gap
      // isn't mistaken for a broken fetch.
      setFetchHint(
        "Call ID, topic ID, and title auto-filled. Paste Expected outcomes and Scope from the portal — they are not available via the portal's public API for current topics.",
      );
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : String(err));
    } finally {
      setFetchingCall(false);
    }
  }, [callUrl]);

  // Guards against a stale generate/refine response landing after the
  // operator has reset the form or started a new run: each mutation of
  // draft state bumps the id, and in-flight responses whose id no
  // longer matches are dropped instead of overwriting fresh state.
  const draftRunIdRef = useRef(0);

  const handleGenerate = useCallback(async () => {
    const errors: string[] = [];
    if (!sectionType) errors.push("section type is required");
    if (!userIntent.trim()) errors.push("user intent is required");
    const topKNum = Number.parseInt(topK, 10);
    if (!Number.isFinite(topKNum) || topKNum < 1 || topKNum > 20) {
      errors.push("top-k examples must be between 1 and 20");
    }
    // AC #1: client-side validation mirrors the server-side Pydantic
    // ge=1 / le=5 constraint so a bad value is caught before the round-trip.
    const maxItNum = Number.parseInt(maxIterations, 10);
    if (
      !Number.isFinite(maxItNum) ||
      maxItNum < MIN_ITERATIONS ||
      maxItNum > MAX_ITERATIONS
    ) {
      errors.push(
        `critic iterations must be between ${MIN_ITERATIONS} and ${MAX_ITERATIONS}`,
      );
    }
    if (errors.length > 0) {
      setClientErrors(errors);
      return;
    }
    setClientErrors([]);
    setServerError(null);
    setGenerating(true);
    // A fresh generate resets the loop state so the operator can start
    // a new accept / refine cycle without stale flags from the prior run.
    setAccepted(false);
    setStoppedAtCap(false);

    const callContext =
      contextMode === "free" ? freeContext : buildStructuredContextString(structured);

    const body: GenerateSectionRequest = {
      section_type: sectionType,
      user_intent: userIntent.trim(),
      call_context: callContext,
      target_programme: targetProgramme === NONE_VALUE ? null : targetProgramme,
      profile_programme: profileProgramme === NONE_VALUE ? null : profileProgramme,
      top_k_examples: topKNum,
      lessons_learned: lessonsLearned,
    };

    const runId = ++draftRunIdRef.current;
    try {
      const result = await generateSection(body);
      if (runId !== draftRunIdRef.current) return;
      setDraft(result);
    } catch (err) {
      if (runId !== draftRunIdRef.current) return;
      setServerError(err instanceof Error ? err.message : String(err));
      setDraft(null);
    } finally {
      if (runId === draftRunIdRef.current) setGenerating(false);
    }
  }, [
    sectionType,
    userIntent,
    topK,
    maxIterations,
    contextMode,
    freeContext,
    structured,
    targetProgramme,
    profileProgramme,
    lessonsLearned,
  ]);

  const handleRefine = useCallback(async () => {
    if (!draft || refining || accepted || stoppedAtCap) return;
    const maxItNum = Number.parseInt(maxIterations, 10);
    if (
      !Number.isFinite(maxItNum) ||
      maxItNum < MIN_ITERATIONS ||
      maxItNum > MAX_ITERATIONS
    ) {
      setServerError(
        `critic iterations must be between ${MIN_ITERATIONS} and ${MAX_ITERATIONS}`,
      );
      return;
    }
    setServerError(null);
    setRefining(true);

    const callContext =
      contextMode === "free" ? freeContext : buildStructuredContextString(structured);
    const topKNum = Number.parseInt(topK, 10) || DEFAULT_TOP_K;

    const body: IterateSectionRequest = {
      section_type: sectionType,
      user_intent: userIntent.trim(),
      call_context: callContext,
      target_programme: targetProgramme === NONE_VALUE ? null : targetProgramme,
      profile_programme: profileProgramme === NONE_VALUE ? null : profileProgramme,
      top_k_examples: topKNum,
      lessons_learned: lessonsLearned,
      max_iterations: maxItNum,
      prior_draft: draft,
    };

    const runId = ++draftRunIdRef.current;
    try {
      const result = await iterateSection(body);
      if (runId !== draftRunIdRef.current) return;
      setDraft(result.draft);
      setStoppedAtCap(result.stopped);
    } catch (err) {
      if (runId !== draftRunIdRef.current) return;
      setServerError(err instanceof Error ? err.message : String(err));
    } finally {
      if (runId === draftRunIdRef.current) setRefining(false);
    }
  }, [
    draft,
    refining,
    accepted,
    stoppedAtCap,
    maxIterations,
    contextMode,
    freeContext,
    structured,
    sectionType,
    userIntent,
    targetProgramme,
    profileProgramme,
    topK,
    lessonsLearned,
  ]);

  // AC #2: "User can stop the loop after any completed iteration."
  // Accept flips the workspace into an immutable state where the
  // Refine button is disabled. The user can still hit Reset to start
  // a brand-new draft.
  const handleAccept = useCallback(() => {
    setAccepted(true);
  }, []);

  function resetForm() {
    // Invalidate any in-flight generate/refine so its late response
    // cannot resurrect the draft we are about to clear.
    draftRunIdRef.current++;
    setGenerating(false);
    setRefining(false);
    setSectionType("");
    setProfileProgramme(NONE_VALUE);
    setTargetProgramme(NONE_VALUE);
    setUserIntent("");
    setFreeContext("");
    setStructured(EMPTY_STRUCTURED);
    setTopK(String(DEFAULT_TOP_K));
    setMaxIterations(String(DEFAULT_MAX_ITERATIONS));
    setLessonsLearned(false);
    setClientErrors([]);
    setServerError(null);
    setDraft(null);
    setAccepted(false);
    setStoppedAtCap(false);
    setCallUrl("");
    setFetchError(null);
    setFetchHint(null);
  }

  // The implicit first pass is iteration 1 (the draft itself); each
  // entry in ``draft.iterations`` is a critic pass. Total iterations
  // currently used = 1 + N records.
  const currentIteration = draft ? 1 + draft.iterations.length : 0;
  const maxItParsed = Number.parseInt(maxIterations, 10);
  const effectiveMaxIterations =
    Number.isFinite(maxItParsed) &&
    maxItParsed >= MIN_ITERATIONS &&
    maxItParsed <= MAX_ITERATIONS
      ? maxItParsed
      : DEFAULT_MAX_ITERATIONS;
  const canRefine =
    draft !== null &&
    !generating &&
    !refining &&
    !accepted &&
    !stoppedAtCap &&
    currentIteration < effectiveMaxIterations;

  // AC #1 of issue #20: which required inputs failed client-side validation,
  // so we can flip aria-invalid on the actual control. Recompute on every
  // render so the announce-on-correction flow works without a separate effect.
  const invalidFields = new Set<string>();
  if (clientErrors.length > 0) {
    if (!sectionType) invalidFields.add("section_type");
    if (!userIntent.trim()) invalidFields.add("user_intent");
    const topKNum = Number.parseInt(topK, 10);
    if (!Number.isFinite(topKNum) || topKNum < 1 || topKNum > 20) {
      invalidFields.add("top_k");
    }
    const maxItNum = Number.parseInt(maxIterations, 10);
    if (
      !Number.isFinite(maxItNum) ||
      maxItNum < MIN_ITERATIONS ||
      maxItNum > MAX_ITERATIONS
    ) {
      invalidFields.add("max_iterations");
    }
  }

  return (
    <section aria-labelledby="drafting-heading">
      <WorkspaceHeader
        eyebrow="Workflow · Draft"
        title="Draft a section"
        description="Pick a section type, point the workflow at one of your drafting profiles, and let the local LLM stitch a first pass from past-proposal evidence."
      />

      <div className="mx-auto w-full max-w-6xl space-y-6 p-6 lg:p-10">
        <h2 id="drafting-heading" className="sr-only">
          Drafting workflow
        </h2>

      {enumsError && (
        <Alert variant="destructive" role="alert">
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
          <AlertTitle>Could not load form choices</AlertTitle>
          <AlertDescription>{enumsError}</AlertDescription>
        </Alert>
      )}
      {profilesError && (
        <Alert variant="destructive" role="alert">
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
          <AlertTitle>Could not load drafting profiles</AlertTitle>
          <AlertDescription>{profilesError}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>1. Configure the draft</CardTitle>
          <CardDescription>
            Required fields are marked with a red asterisk. Empty values are rejected
            before the request is sent.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6" aria-busy={generating || refining}>
          {clientErrors.length > 0 && (
            <Alert variant="destructive" role="alert">
              <AlertCircle className="h-4 w-4" aria-hidden="true" />
              <AlertTitle>Please complete the required fields</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-5">
                  {clientErrors.map((err) => (
                    <li key={err}>{err}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="section_type">
                Section type <RequiredMark />
              </Label>
              <Select value={sectionType} onValueChange={setSectionType}>
                <SelectTrigger
                  id="section_type"
                  aria-required="true"
                  aria-invalid={invalidFields.has("section_type")}
                >
                  <SelectValue placeholder="Select a section" />
                </SelectTrigger>
                <SelectContent>
                  {sectionTypes.map((s) => (
                    <SelectItem key={s} value={s}>
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="profile_programme">Drafting profile</Label>
              <Select
                value={profileProgramme}
                onValueChange={setProfileProgramme}
              >
                <SelectTrigger id="profile_programme">
                  <SelectValue placeholder="Use generic guidance" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>Generic (no profile)</SelectItem>
                  {profileItems.map((p) => (
                    <SelectItem key={p.programme} value={p.programme}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Pick a programme-specific profile to override the default section guidance.
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Call / topic context</Label>

            {/* Issue #67: paste a portal URL → auto-fill structured tab. */}
            <div className="rounded-md border border-input bg-muted/30 p-3 space-y-2">
              <Label htmlFor="call_url" className="text-sm font-medium">
                Auto-fill from EU Funding & Tenders Portal URL
              </Label>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Input
                  id="call_url"
                  type="url"
                  placeholder="https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-..."
                  value={callUrl}
                  onChange={(e) => setCallUrl(e.target.value)}
                  disabled={fetchingCall}
                  aria-describedby="call_url_help"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => void handleFetchCall()}
                  disabled={fetchingCall || !callUrl.trim()}
                  aria-label="Fetch call context from this URL"
                >
                  {fetchingCall ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                      Fetching…
                    </>
                  ) : (
                    <>
                      <Download className="h-4 w-4" aria-hidden="true" />
                      Fetch
                    </>
                  )}
                </Button>
              </div>
              <p id="call_url_help" className="text-xs text-muted-foreground">
                Contacts <code>ec.europa.eu</code> and{" "}
                <code>api.tech.ec.europa.eu</code> to recover the call ID, topic ID,
                and topic title for the pasted URL.
              </p>
              {fetchError && (
                <Alert variant="destructive" role="alert">
                  <AlertCircle className="h-4 w-4" aria-hidden="true" />
                  <AlertTitle>Could not fetch call context</AlertTitle>
                  <AlertDescription>{fetchError}</AlertDescription>
                </Alert>
              )}
              {fetchHint && !fetchError && (
                <Alert role="status" aria-live="polite">
                  <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                  <AlertTitle>Auto-fill complete</AlertTitle>
                  <AlertDescription>{fetchHint}</AlertDescription>
                </Alert>
              )}
            </div>

            <Tabs
              value={contextMode}
              onValueChange={(v) => setContextMode(v as ContextMode)}
            >
              <TabsList>
                <TabsTrigger value="free">Free text</TabsTrigger>
                <TabsTrigger value="structured">Structured</TabsTrigger>
              </TabsList>
              <TabsContent value="free">
                <Textarea
                  id="free_context"
                  placeholder="Paste the call summary, topic blurb, or any context the LLM should ground on (optional)."
                  className="min-h-[120px]"
                  value={freeContext}
                  onChange={(e) => setFreeContext(e.target.value)}
                />
              </TabsContent>
              <TabsContent value="structured" className="space-y-3">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="structured_call_id">Call ID</Label>
                    <Input
                      id="structured_call_id"
                      placeholder="HORIZON-CL3-2024-CS-01"
                      value={structured.callId}
                      onChange={(e) =>
                        setStructured((s) => ({ ...s, callId: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="structured_topic_id">Topic ID</Label>
                    <Input
                      id="structured_topic_id"
                      placeholder="883588"
                      value={structured.topicId}
                      onChange={(e) =>
                        setStructured((s) => ({ ...s, topicId: e.target.value }))
                      }
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="structured_topic_title">Topic title</Label>
                  <Input
                    id="structured_topic_title"
                    placeholder="Resilient digital infrastructure for critical sectors"
                    value={structured.topicTitle}
                    onChange={(e) =>
                      setStructured((s) => ({ ...s, topicTitle: e.target.value }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="structured_outcomes">Expected outcomes</Label>
                  <Textarea
                    id="structured_outcomes"
                    placeholder="- Outcome one&#10;- Outcome two"
                    className="min-h-[80px]"
                    value={structured.expectedOutcomes}
                    onChange={(e) =>
                      setStructured((s) => ({
                        ...s,
                        expectedOutcomes: e.target.value,
                      }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="structured_scope">Scope</Label>
                  <Textarea
                    id="structured_scope"
                    placeholder="What this topic covers (free text)."
                    className="min-h-[80px]"
                    value={structured.scope}
                    onChange={(e) =>
                      setStructured((s) => ({ ...s, scope: e.target.value }))
                    }
                  />
                </div>
              </TabsContent>
            </Tabs>
          </div>

          <div className="space-y-2">
            <Label htmlFor="user_intent">
              User intent / bullets <RequiredMark />
            </Label>
            <Textarea
              id="user_intent"
              placeholder="What should this section communicate? One or two sentences, or a short list of bullets."
              className="min-h-[100px]"
              value={userIntent}
              onChange={(e) => setUserIntent(e.target.value)}
              aria-required="true"
              aria-invalid={invalidFields.has("user_intent")}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="target_programme">Programme filter</Label>
              <Select value={targetProgramme} onValueChange={setTargetProgramme}>
                <SelectTrigger id="target_programme">
                  <SelectValue placeholder="All programmes" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>All programmes</SelectItem>
                  {programmes.map((p) => (
                    <SelectItem key={p} value={p}>
                      {p}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="top_k">Top-k examples</Label>
              <Input
                id="top_k"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                aria-invalid={invalidFields.has("top_k")}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="max_iterations">Critic iterations</Label>
              <Input
                id="max_iterations"
                type="number"
                min={MIN_ITERATIONS}
                max={MAX_ITERATIONS}
                value={maxIterations}
                onChange={(e) => setMaxIterations(e.target.value)}
                aria-describedby="max_iterations_help"
                aria-invalid={invalidFields.has("max_iterations")}
              />
              <p id="max_iterations_help" className="text-xs text-muted-foreground">
                {MIN_ITERATIONS}-{MAX_ITERATIONS}. Default 3. Use the Refine
                button below to step through critic passes; stop any time.
              </p>
            </div>

            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-input focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  checked={lessonsLearned}
                  onChange={(e) => setLessonsLearned(e.target.checked)}
                />
                <span>Surface rejected examples (lessons learned)</span>
              </label>
            </div>
          </div>

          {serverError && (
            <Alert variant="destructive" role="alert">
              <AlertCircle className="h-4 w-4" aria-hidden="true" />
              <AlertTitle>Generation failed</AlertTitle>
              <AlertDescription>{serverError}</AlertDescription>
            </Alert>
          )}

          {generating && (
            <p role="status" aria-live="polite" className="sr-only">
              Generating draft, please wait.
            </p>
          )}

          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={resetForm}
            >
              Reset
            </Button>
            <Button
              type="button"
              variant="amber"
              onClick={() => void handleGenerate()}
              disabled={generating || refining}
            >
              {generating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  Generating…
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  Generate draft
                </>
              )}
            </Button>
          </div>

          {/* Hidden tag so a maintainer searching for the AC #1 list lands on this file. */}
          <span data-required-fields={REQUIRED_FIELDS.join(",")} className="sr-only" />
        </CardContent>
      </Card>

      {draft && (
        <section
          aria-labelledby="generated-draft-heading"
          aria-busy={refining || generating}
          className="space-y-4"
        >
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2
                id="generated-draft-heading"
                className="font-display text-2xl tracking-tight text-brand-navy"
              >
                Review the draft
              </h2>
              <div className="mt-2 flex flex-wrap items-center gap-2" aria-live="polite">
                {Array.from({ length: effectiveMaxIterations }).map((_, i) => {
                  const done = i + 1 <= currentIteration;
                  return (
                    <span
                      key={i}
                      aria-hidden="true"
                      className={cn(
                        "h-1.5 w-7 rounded-full transition-colors",
                        done ? "bg-brand-amber" : "bg-brand-navy/10",
                      )}
                    />
                  );
                })}
                <span className="ml-1 text-sm text-brand-navy/70">
                  Iteration {currentIteration} of {effectiveMaxIterations}
                  {accepted && " — accepted"}
                  {stoppedAtCap && !accepted && " — iteration cap reached"}
                  {generating && " — regenerating, preview below is the previous draft"}
                </span>
              </div>
            </div>
            {/* AC #2 controls: Refine runs one more critic pass; Accept
                stops the loop. Both disappear once the user accepts
                so the workspace state is unambiguous. */}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleAccept}
                disabled={accepted || refining || generating}
                aria-label="Accept this draft and stop the critic loop"
              >
                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                {accepted ? "Accepted" : "Accept draft"}
              </Button>
              <Button
                type="button"
                variant="amber"
                onClick={() => void handleRefine()}
                disabled={!canRefine}
                aria-label="Run one more critic loop iteration"
              >
                {refining ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Refining…
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                    Refine ({currentIteration}/{effectiveMaxIterations})
                  </>
                )}
              </Button>
            </div>
          </div>
          {stoppedAtCap && !accepted && (
            <Alert role="status" aria-live="polite">
              <AlertCircle className="h-4 w-4" aria-hidden="true" />
              <AlertTitle>Iteration cap reached</AlertTitle>
              <AlertDescription>
                You have used all {effectiveMaxIterations} configured critic
                iterations. Raise the cap above and regenerate to keep refining,
                or accept this draft.
              </AlertDescription>
            </Alert>
          )}
          <div
            className={cn("transition-opacity", generating && "opacity-50")}
            title={generating ? "Regenerating — this preview still shows the previous draft" : undefined}
          >
            <DraftPreview draft={draft} />
          </div>
        </section>
      )}
      </div>
    </section>
  );
}

function RequiredMark() {
  return (
    <span aria-hidden="true" className="text-destructive">
      *
    </span>
  );
}
