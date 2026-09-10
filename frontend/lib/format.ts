/** Number formatting shared by the estimate hero, tables and chart axes. */

export function sig(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e7)) return value.toExponential(2);
  const decimals = abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 1 ? 3 : digits;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  });
}

export function pval(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

export function signed(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return (value > 0 ? "+" : "") + sig(value, digits);
}

export function count(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toLocaleString();
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Verdict -> the CSS custom properties defined in globals.css. */
export const verdictColor = {
  pass: { fg: "var(--pass)", bg: "var(--pass-soft)", label: "Passes" },
  warn: { fg: "var(--warn)", bg: "var(--warn-soft)", label: "Caution" },
  fail: { fg: "var(--fail)", bg: "var(--fail-soft)", label: "Fails" },
  info: { fg: "var(--info)", bg: "var(--info-soft)", label: "Context" },
} as const;
