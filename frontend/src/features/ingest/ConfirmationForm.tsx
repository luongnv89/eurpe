import { useEffect, useState } from "react";

import { AlertCircle, Loader2 } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import type {
  ConfirmRequest,
  EnumsResponse,
  ParseResponse,
} from "./api";

// All required fields are listed up front so the AC #2 invariant
// ("UI requires programme, call/topic, and source-status confirmation")
// is one diff line away from review.
const REQUIRED_FIELDS = ["programme", "call_id", "topic_id", "year", "outcome"] as const;

// Hoisted out of the component so a re-render doesn't recompute the value
// and accidentally retrigger the field-reset effect at year boundaries.
const DEFAULT_YEAR = new Date().getFullYear();

interface Props {
  parseResult: ParseResponse;
  enums: EnumsResponse;
  submitting: boolean;
  errorMessage: string | null;
  onSubmit: (body: ConfirmRequest) => void;
  onCancel: () => void;
}

export function ConfirmationForm({
  parseResult,
  enums,
  submitting,
  errorMessage,
  onSubmit,
  onCancel,
}: Props) {
  const suggested = parseResult.suggested ?? {};

  const [programme, setProgramme] = useState<string>(suggested.programme ?? "");
  const [callId, setCallId] = useState<string>(suggested.call_id ?? "");
  const [topicId, setTopicId] = useState<string>(suggested.topic_id ?? "");
  const [year, setYear] = useState<string>(String(DEFAULT_YEAR));
  const [outcome, setOutcome] = useState<string>("");
  const [proposalTitle, setProposalTitle] = useState<string>(
    suggested.proposal_title ?? parseResult.title ?? "",
  );
  const [consortium, setConsortium] = useState<string>("");
  const [clientErrors, setClientErrors] = useState<string[]>([]);

  // Keep the local fields in sync if the parent sends a fresh
  // parseResult (e.g. operator started over). The dep list keys on the
  // parse_token because every other field on parseResult is downstream
  // of it — a new token means a brand-new upload.
  useEffect(() => {
    setProgramme(suggested.programme ?? "");
    setCallId(suggested.call_id ?? "");
    setTopicId(suggested.topic_id ?? "");
    setProposalTitle(suggested.proposal_title ?? parseResult.title ?? "");
    setConsortium("");
    setOutcome("");
    setYear(String(DEFAULT_YEAR));
    setClientErrors([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parseResult.parse_token]);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const errors: string[] = [];
    if (!programme) errors.push("programme is required");
    if (!callId.trim()) errors.push("call_id is required");
    if (!topicId.trim()) errors.push("topic_id is required");
    if (!outcome) errors.push("outcome is required");
    const yearNum = Number.parseInt(year, 10);
    if (!Number.isFinite(yearNum) || yearNum < 2014 || yearNum > 2099) {
      errors.push("year must be between 2014 and 2099");
    }
    if (errors.length > 0) {
      setClientErrors(errors);
      return;
    }
    setClientErrors([]);
    const body: ConfirmRequest = {
      parse_token: parseResult.parse_token,
      programme,
      call_id: callId.trim(),
      topic_id: topicId.trim(),
      year: yearNum,
      outcome,
      proposal_title: proposalTitle.trim() || null,
      consortium_acronym: consortium.trim() || null,
    };
    onSubmit(body);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
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
      {errorMessage && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Server rejected the submission</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="programme">
            Programme <RequiredMark />
          </Label>
          <Select value={programme} onValueChange={setProgramme}>
            <SelectTrigger id="programme">
              <SelectValue placeholder="Select programme" />
            </SelectTrigger>
            <SelectContent>
              {enums.programme.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Hint suggestedValue={suggested.programme} actualValue={programme} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="outcome">
            Outcome (source-status) <RequiredMark />
          </Label>
          <Select value={outcome} onValueChange={setOutcome}>
            <SelectTrigger id="outcome">
              <SelectValue placeholder="Select outcome" />
            </SelectTrigger>
            <SelectContent>
              {enums.source_status.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Drives every chunk's <code>source_status</code>; cannot be changed after indexing.
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="call_id">
            Call ID <RequiredMark />
          </Label>
          <Input
            id="call_id"
            value={callId}
            onChange={(e) => setCallId(e.target.value)}
            placeholder="HORIZON-CL5-2024-D3-02"
            autoComplete="off"
          />
          <Hint suggestedValue={suggested.call_id} actualValue={callId} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="topic_id">
            Topic ID <RequiredMark />
          </Label>
          <Input
            id="topic_id"
            value={topicId}
            onChange={(e) => setTopicId(e.target.value)}
            placeholder="HORIZON-CL5-2024-D3-02-01"
            autoComplete="off"
          />
          <Hint suggestedValue={suggested.topic_id} actualValue={topicId} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="year">
            Year <RequiredMark />
          </Label>
          <Input
            id="year"
            type="number"
            min={2014}
            max={2099}
            value={year}
            onChange={(e) => setYear(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="proposal_title">Proposal title</Label>
          <Input
            id="proposal_title"
            value={proposalTitle}
            onChange={(e) => setProposalTitle(e.target.value)}
            placeholder="Optional"
          />
        </div>

        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="consortium_acronym">Consortium acronym</Label>
          <Input
            id="consortium_acronym"
            value={consortium}
            onChange={(e) => setConsortium(e.target.value)}
            placeholder="Optional, e.g. STP"
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-2 pt-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Indexing…
            </>
          ) : (
            "Confirm and index"
          )}
        </Button>
      </div>

      {/* Hidden tag so a maintainer searching for the AC #2 list lands on this file. */}
      <span data-required-fields={REQUIRED_FIELDS.join(",")} className="sr-only" />
    </form>
  );
}

function RequiredMark() {
  return <span aria-hidden="true" className="text-destructive">*</span>;
}

function Hint({
  suggestedValue,
  actualValue,
}: {
  suggestedValue?: string;
  actualValue: string;
}) {
  if (!suggestedValue) return null;
  if (suggestedValue === actualValue) {
    return (
      <p className="text-xs text-muted-foreground">
        Inferred from filename ·{" "}
        <code className="rounded bg-muted px-1 py-0.5">{suggestedValue}</code>
      </p>
    );
  }
  return (
    <p className="text-xs text-muted-foreground">
      Server suggested{" "}
      <code className="rounded bg-muted px-1 py-0.5">{suggestedValue}</code> — overridden.
    </p>
  );
}
