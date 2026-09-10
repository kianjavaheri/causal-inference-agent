import Link from "next/link";

export const metadata = {
  title: "About · Causal Inference Agent",
  description: "What this project does, and what a causal estimate can and cannot tell you.",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="mb-3 text-[13px] font-medium uppercase tracking-wider text-ink-3">
        {title}
      </h2>
      <div className="space-y-4 text-[15px] leading-[1.75] text-ink-2">{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <main className="mx-auto max-w-3xl px-5 py-12 sm:px-8 sm:py-16">
      <h1 className="text-[30px] font-semibold leading-tight tracking-tight text-ink sm:text-[36px]">
        About this project
      </h1>
      <p className="mt-4 text-[16.5px] leading-relaxed text-ink-2">
        Give it a dataset and a question in plain English. It works out which causal design
        the data can actually support, runs it, and checks the assumptions that design
        depends on.
      </p>

      <div className="mt-12">
        <Section title="The problem it solves">
          <p>
            Most tools will happily compute a difference between two groups and call it an
            effect. The hard part of causal inference is not the arithmetic — it is knowing
            which comparison is <em>legitimate</em> given how the data came to be, and being
            honest about what that comparison cannot tell you.
          </p>
          <p>
            So the interesting behaviour here is not that it produces a number. It is that it
            declines to produce the wrong one, and says why.
          </p>
        </Section>

        <Section title="How it works">
          <p>
            Four steps, streamed live so you can watch it reason:
          </p>
          <ol className="ml-1 space-y-2.5">
            {[
              ["Profile", "Read the CSV's structure — column types, missingness, whether it is a panel, which columns could plausibly be an outcome, a treatment, a unit, a time period."],
              ["Plan", "Test all five designs against that structure. Decide which are identifiable, assign columns to roles, and justify the choice against the question asked."],
              ["Execute", "Run the estimator, and run every assumption check that design permits."],
              ["Report", "Write up the result, separating what the diagnostics showed from what remains an act of faith."],
            ].map(([name, body], i) => (
              <li key={name} className="flex gap-3">
                <span className="tnum mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-surface-2 text-[11px] font-semibold text-ink-3">
                  {i + 1}
                </span>
                <span>
                  <span className="font-medium text-ink">{name}.</span> {body}
                </span>
              </li>
            ))}
          </ol>
        </Section>

        <Section title="The planner is not just a language model">
          <p>
            A deterministic identifiability engine examines the data&apos;s structure and
            decides which designs are possible, which columns fill each role, and why the
            rest are blocked. A language model then chooses among the feasible options in
            light of your question and writes the justification.
          </p>
          <p>
            The model cannot make an unidentifiable design feasible. If it names a blocked
            one, the engine overrides it and says so in the reasoning trail. With no API key
            configured, the engine runs alone — and still picks the right design on every
            demo dataset.
          </p>
        </Section>

        <Section title="What it will not do">
          <p>
            It refuses to invent an instrument from correlation — a variable that predicts
            treatment is just as likely a confounder. It rejects a column that merely marks
            the post period when asked to treat it as a treatment. It refuses to match when a
            covariate secretly encodes the assignment rule. And when nothing works, it says
            what is missing rather than returning a number anyway.
          </p>
        </Section>

        <Section title="A note on what any of this can tell you">
          <p>
            Every design here identifies a causal effect only under assumptions the data can
            never fully verify. The diagnostics test the <em>observable implications</em> of
            those assumptions — parallel pre-trends, not parallel trends; covariate balance,
            not unconfoundedness; instrument strength, not exclusion.
          </p>
          <p className="text-ink">
            A green check means the design is not obviously broken. It does not mean the
            estimate is true. The reports are written to say so.
          </p>
        </Section>

        <Section title="Built with">
          <p>
            FastAPI and a hand-rolled estimation layer on numpy/scipy — no black-box causal
            package, so every standard error and diagnostic is inspectable. Next.js and
            Recharts on the front. Estimators are validated against datasets with known
            planted effects; the test suite asserts each confidence interval covers the truth.
          </p>
        </Section>
      </div>

      <div className="mt-4 flex flex-wrap gap-3 border-t border-border-base pt-8">
        <Link
          href="/"
          className="cursor-pointer rounded-lg px-4 py-2 text-[13.5px] font-medium text-white transition-opacity hover:opacity-90"
          style={{ background: "var(--accent)" }}
        >
          Try it on a dataset
        </Link>
        <Link
          href="/methods"
          className="cursor-pointer rounded-lg border border-border-strong px-4 py-2 text-[13.5px] font-medium text-ink-2 transition-colors hover:border-accent hover:text-accent"
        >
          The five designs
        </Link>
      </div>
    </main>
  );
}
