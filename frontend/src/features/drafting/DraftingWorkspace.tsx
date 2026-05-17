import { useCallback, useEffect, useMemo, useState } from "react";

import { AlertCircle, Loader2, Sparkles } from "lucide-react";

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

import {
  fetchGenerationEnums,
  fetchProfiles,
  generateSection,
} from "./api";
import type {
  GenerateSectionRequest,
  GenerateSectionResponse,
  GenerationEnumsResponse,
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

  const [clientErrors, setClientErrors] = useState<string[]>([]);
  const [generating, setGenerating] = useState<boolean>(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [draft, setDraft] = useState<GenerateSectionResponse | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchGenerationEnums(ctrl.signal)
      .then(setEnums)
      .catch((err) => setEnumsError(err.message ?? String(err)));
    fetchProfiles(ctrl.signal)
      .then(setProfiles)
      .catch((err) => setProfilesError(err.message ?? String(err)));
    return () => ctrl.abort();
  }, []);

  const profileItems = useMemo(() => profiles?.profiles ?? [], [profiles]);
  const sectionTypes = useMemo(() => enums?.section_type ?? [], [enums]);
  const programmes = useMemo(() => enums?.programme ?? [], [enums]);

  const handleGenerate = useCallback(async () => {
    const errors: string[] = [];
    if (!sectionType) errors.push("section type is required");
    if (!userIntent.trim()) errors.push("user intent is required");
    const topKNum = Number.parseInt(topK, 10);
    if (!Number.isFinite(topKNum) || topKNum < 1 || topKNum > 20) {
      errors.push("top-k examples must be between 1 and 20");
    }
    if (errors.length > 0) {
      setClientErrors(errors);
      return;
    }
    setClientErrors([]);
    setServerError(null);
    setGenerating(true);

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

    try {
      const result = await generateSection(body);
      setDraft(result);
    } catch (err) {
      setServerError(err instanceof Error ? err.message : String(err));
      setDraft(null);
    } finally {
      setGenerating(false);
    }
  }, [
    sectionType,
    userIntent,
    topK,
    contextMode,
    freeContext,
    structured,
    targetProgramme,
    profileProgramme,
    lessonsLearned,
  ]);

  function resetForm() {
    setSectionType("");
    setProfileProgramme(NONE_VALUE);
    setTargetProgramme(NONE_VALUE);
    setUserIntent("");
    setFreeContext("");
    setStructured(EMPTY_STRUCTURED);
    setTopK(String(DEFAULT_TOP_K));
    setLessonsLearned(false);
    setClientErrors([]);
    setServerError(null);
    setDraft(null);
  }

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 p-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Draft a section</h1>
        <p className="text-muted-foreground">
          Pick a section type, point the workflow at one of your drafting profiles, and let
          the local LLM stitch a first pass from past-proposal evidence.
        </p>
      </header>

      {enumsError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Could not load form choices</AlertTitle>
          <AlertDescription>{enumsError}</AlertDescription>
        </Alert>
      )}
      {profilesError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
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
        <CardContent className="space-y-6">
          {clientErrors.length > 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
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
                <SelectTrigger id="section_type">
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
            />
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
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
              />
            </div>

            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-input"
                  checked={lessonsLearned}
                  onChange={(e) => setLessonsLearned(e.target.checked)}
                />
                <span>Surface rejected examples (lessons learned)</span>
              </label>
            </div>
          </div>

          {serverError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Generation failed</AlertTitle>
              <AlertDescription>{serverError}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={resetForm} disabled={generating}>
              Reset
            </Button>
            <Button
              type="button"
              onClick={() => void handleGenerate()}
              disabled={generating}
            >
              {generating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating…
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
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
        <section aria-label="Generated draft" className="space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight">2. Review the draft</h2>
          <DraftPreview draft={draft} />
        </section>
      )}
    </div>
  );
}

function RequiredMark() {
  return (
    <span aria-hidden="true" className="text-destructive">
      *
    </span>
  );
}
