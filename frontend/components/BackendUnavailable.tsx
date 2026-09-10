"use client";

import { API_BASE } from "@/lib/api";

/**
 * Three different people can land on a dead backend, and each needs a different
 * message:
 *
 *  - local:        a developer running `next dev` who hasn't started the API.
 *  - unconfigured: a production build that still points at localhost. The build was
 *                  made without NEXT_PUBLIC_API_BASE, so no amount of retrying helps —
 *                  only the site's owner can fix it, by setting the variable and
 *                  redeploying. This is only ever seen before a deploy is finished.
 *  - unreachable:  a correctly configured site whose API is down or still waking up.
 *                  The visitor's only useful move is to try again.
 *
 * NODE_ENV and NEXT_PUBLIC_API_BASE are both inlined at build time, so this is decided
 * once per build rather than guessed at from the browser.
 */
type Cause = "local" | "unconfigured" | "unreachable";

const LOCALHOST = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/;

function diagnose(): Cause {
  if (process.env.NODE_ENV !== "production") return "local";
  return LOCALHOST.test(API_BASE) ? "unconfigured" : "unreachable";
}

const COPY: Record<Cause, { tone: "warn" | "fail"; title: string; retry: boolean }> = {
  local: {
    tone: "warn",
    title: "The backend isn't running.",
    retry: true,
  },
  unconfigured: {
    tone: "fail",
    title: "This deployment isn't connected to its analysis service yet.",
    retry: false,
  },
  unreachable: {
    tone: "warn",
    title: "The analysis service isn't responding.",
    retry: true,
  },
};

export function BackendUnavailable({ className = "" }: { className?: string }) {
  const cause = diagnose();
  const { tone, title, retry } = COPY[cause];

  return (
    <div
      role="alert"
      className={`rounded-xl px-5 py-4 text-[13.5px] leading-relaxed ${className}`}
      style={{ color: `var(--${tone})`, background: `var(--${tone}-soft)` }}
    >
      <p className="font-medium">{title}</p>

      <p className="mt-1 text-ink-2">
        {cause === "local" ? (
          <>
            Start it in another terminal with{" "}
            <code className="font-mono text-[12.5px]">
              cd backend &amp;&amp; .venv/bin/uvicorn app.main:app --reload
            </code>
            , then try again.
          </>
        ) : cause === "unconfigured" ? (
          <>
            The site was built without a backend address, so it is looking for one on
            this computer. Set{" "}
            <code className="font-mono text-[12.5px]">NEXT_PUBLIC_API_BASE</code> to the
            backend&apos;s URL in the hosting settings and redeploy — the address is fixed
            when the site is built, so changing the setting alone won&apos;t take effect.
          </>
        ) : (
          <>
            It may be starting up after a quiet spell, which usually takes under a minute.
          </>
        )}
      </p>

      {retry ? (
        <button
          onClick={() => window.location.reload()}
          className="mt-3 cursor-pointer rounded-lg border px-3 py-1.5 text-[12.5px] font-medium transition-opacity hover:opacity-75"
          style={{ borderColor: `var(--${tone})`, color: `var(--${tone})` }}
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}
