"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSyncExternalStore } from "react";

const LINKS = [
  { href: "/", label: "Agent" },
  { href: "/about", label: "About" },
  { href: "/methods", label: "Methods" },
];

type Theme = "light" | "dark";

/**
 * The active theme lives outside React: an inline script in <head> applies the saved
 * choice before first paint, and the OS supplies a default. useSyncExternalStore reads
 * that source of truth directly, which keeps the server render (null) from disagreeing
 * with the client and avoids a hydration mismatch.
 */
function subscribe(onChange: () => void): () => void {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", onChange);
  window.addEventListener("themechange", onChange);
  return () => {
    media.removeEventListener("change", onChange);
    window.removeEventListener("themechange", onChange);
  };
}

function getSnapshot(): Theme {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "light" || attr === "dark") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Nothing is known about the viewer's theme on the server. */
function getServerSnapshot(): Theme | null {
  return null;
}

/** Glass pill navbar, floating over the page. */
export function NavBar() {
  const pathname = usePathname();
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* private mode: the choice just won't persist */
    }
    window.dispatchEvent(new Event("themechange"));
  };

  return (
    <div className="sticky top-0 z-50 px-4 pt-4 sm:px-6">
      <nav
        className="mx-auto flex max-w-4xl items-center gap-4 rounded-full border border-border-base px-5 py-3 shadow-[0_4px_24px_rgb(0_0_0/0.06)] backdrop-blur-xl"
        style={{ background: "color-mix(in srgb, var(--surface) 72%, transparent)" }}
      >
        <Link
          href="/"
          className="cursor-pointer text-[14.5px] font-semibold tracking-tight text-ink transition-opacity hover:opacity-70"
        >
          Causal Inference Agent
        </Link>

        <div className="ml-auto flex items-center gap-1 sm:gap-2">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`cursor-pointer rounded-full px-3 py-1.5 text-[13.5px] transition-colors ${
                  active
                    ? "bg-surface-2 font-medium text-ink"
                    : "text-ink-2 hover:text-ink"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
          <button
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="cursor-pointer rounded-full px-3 py-1.5 text-[13.5px] text-ink-3 transition-colors hover:text-ink"
          >
            {theme === null ? "" : theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </nav>
    </div>
  );
}
