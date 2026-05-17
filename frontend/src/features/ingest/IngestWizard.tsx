import { useCallback, useEffect, useState } from "react";

import { AlertCircle, CheckCircle2, FileText, Loader2, Upload } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import {
  confirmIngestion,
  fetchEnums,
  uploadForParse,
} from "./api";
import type {
  ConfirmRequest,
  ConfirmResponse,
  EnumsResponse,
  ParseResponse,
} from "./api";
import { ConfirmationForm } from "./ConfirmationForm";

type Step = "upload" | "confirm" | "success";

/**
 * Three-step proposal ingestion flow:
 *
 *   upload  → POST /api/ingestion/parse → ParseResponse
 *   confirm → POST /api/ingestion/confirm → ConfirmResponse
 *   success → recap of what landed on disk + "Ingest another"
 *
 * Enum vocabularies (programme + source_status) are fetched once on
 * mount so the closed-set Selects can never type-drift away from the
 * Python enums.
 */
export function IngestWizard() {
  const [step, setStep] = useState<Step>("upload");
  const [enums, setEnums] = useState<EnumsResponse | null>(null);
  const [enumsError, setEnumsError] = useState<string | null>(null);
  const [parseResult, setParseResult] = useState<ParseResponse | null>(null);
  const [confirmResult, setConfirmResult] = useState<ConfirmResponse | null>(null);
  const [parsing, setParsing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchEnums(ctrl.signal)
      .then(setEnums)
      .catch((err) => setEnumsError(err.message ?? String(err)));
    return () => ctrl.abort();
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setErrorMessage(null);
    setParsing(true);
    try {
      const result = await uploadForParse(file);
      setParseResult(result);
      setStep("confirm");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setErrorMessage(message);
    } finally {
      setParsing(false);
    }
  }, []);

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  }

  async function handleConfirm(body: ConfirmRequest) {
    setErrorMessage(null);
    setSubmitting(true);
    try {
      const result = await confirmIngestion(body);
      setConfirmResult(result);
      setStep("success");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setStep("upload");
    setParseResult(null);
    setConfirmResult(null);
    setErrorMessage(null);
  }

  return (
    <section
      aria-labelledby="ingest-heading"
      className="mx-auto w-full max-w-3xl space-y-6 p-6"
    >
      <header className="space-y-2">
        <h1 id="ingest-heading" className="text-3xl font-semibold tracking-tight">
          Ingest a proposal
        </h1>
        <p className="text-muted-foreground">
          Drop a proposal PDF, review the extracted metadata, and add it to the local index.
        </p>
      </header>

      {enumsError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
          <AlertTitle>Could not load form choices</AlertTitle>
          <AlertDescription>{enumsError}</AlertDescription>
        </Alert>
      )}

      {step === "upload" && (
        <Card>
          <CardHeader>
            <CardTitle>1. Upload a PDF</CardTitle>
            <CardDescription>
              The file is parsed locally with Docling. Nothing leaves your machine.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* role="presentation" is intentional: the visible "Choose file"
                Button below is the keyboard-accessible entry point. The
                drag-and-drop is a pointer-only enhancement (WCAG SC 2.5.7
                exemption — equivalent pointerless path provided). */}
            <div
              role="presentation"
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              aria-busy={parsing}
              className={`flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-10 text-center transition-colors ${
                dragActive ? "border-primary bg-accent" : "border-muted"
              }`}
            >
              {parsing ? (
                <div role="status" aria-live="polite" className="flex flex-col items-center gap-3">
                  <Loader2
                    className="h-8 w-8 animate-spin text-muted-foreground"
                    aria-hidden="true"
                  />
                  <p className="text-sm text-muted-foreground">Parsing PDF…</p>
                </div>
              ) : (
                <>
                  <Upload className="h-8 w-8 text-muted-foreground" aria-hidden="true" />
                  <p className="text-sm text-muted-foreground">Drop a PDF here or pick one</p>
                  <label className="cursor-pointer">
                    <span className="sr-only">Choose a PDF file to upload</span>
                    <input
                      type="file"
                      accept="application/pdf,.pdf"
                      className="hidden"
                      onChange={onPickFile}
                    />
                    <Button asChild variant="outline">
                      <span>Choose file</span>
                    </Button>
                  </label>
                </>
              )}
            </div>
            {errorMessage && (
              <Alert className="mt-4" variant="destructive">
                <AlertCircle className="h-4 w-4" aria-hidden="true" />
                <AlertTitle>Upload failed</AlertTitle>
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {step === "confirm" && parseResult && enums && (
        <Card>
          <CardHeader>
            <CardTitle>2. Confirm metadata</CardTitle>
            <CardDescription>
              Review the extracted values, fill in anything missing, then index this proposal.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex items-center gap-2 rounded-md border bg-muted px-3 py-2 text-sm">
              <FileText className="h-4 w-4" aria-hidden="true" />
              <span className="truncate" title={parseResult.source_path}>
                {parseResult.title ?? parseResult.source_path}
              </span>
              <span className="ml-auto text-xs text-muted-foreground">
                {parseResult.section_count} sections · {parseResult.page_count ?? "?"} pages
              </span>
            </div>
            <ConfirmationForm
              parseResult={parseResult}
              enums={enums}
              submitting={submitting}
              errorMessage={errorMessage}
              onSubmit={handleConfirm}
              onCancel={reset}
            />
          </CardContent>
        </Card>
      )}

      {step === "success" && confirmResult && (
        <Card
          role="status"
          aria-live="polite"
          aria-label={`Proposal indexed — ${confirmResult.chunks_added} chunks added to ${confirmResult.collection}`}
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-green-700" aria-hidden="true" />
              Proposal indexed
            </CardTitle>
            <CardDescription>
              Added <strong>{confirmResult.chunks_added}</strong> chunks to the{" "}
              <code>{confirmResult.collection}</code> collection.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <p className="text-muted-foreground">YAML sidecar persisted to</p>
              <code className="block rounded bg-muted px-2 py-1 text-xs break-all">
                {confirmResult.sidecar_path}
              </code>
            </div>
            <Button onClick={reset}>Ingest another</Button>
          </CardContent>
        </Card>
      )}
    </section>
  );
}
