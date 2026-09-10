# Frontend

Next.js UI for the Causal Inference Research Agent. See the [root README](../README.md)
for what the project does and how to run it.

```bash
npm install
npm run dev
```

Point it at a backend with `NEXT_PUBLIC_API_BASE` (see `.env.example`); it defaults to
`http://127.0.0.1:8000`.

- `app/page.tsx` — the four-step flow
- `components/ReasoningTrail.tsx` — the streamed agent timeline
- `components/diagnostics/` — one component per diagnostic payload kind
- `lib/api.ts` — REST calls and the SSE reader for `/run`
