import * as React from "react";

import { cn } from "@/lib/utils";

// Stock shadcn slate-base Textarea. Multi-line text input used by the
// drafting workspace for ``user intent`` (the freeform sentence or two
// that drives retrieval) and the pasted call/topic context block. Kept
// thin on purpose — sizing comes from a parent ``className`` so the
// drafting workspace can grow it taller without forking the component.
const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => {
  return (
    <textarea
      ref={ref}
      className={cn(
        "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
});
Textarea.displayName = "Textarea";

export { Textarea };
