# UI Layer — Claude Code Context

## Purpose

Next.js 14 App Router frontend. Provides the chat interface, book discovery, and shelf management. Streams AI responses token-by-token via SSE.

## Directory Layout

```
ui/
├── app/
│   ├── layout.tsx              # Root layout: fonts, providers, auth
│   ├── page.tsx                # Home → redirects to /chat
│   ├── (chat)/
│   │   ├── layout.tsx          # Chat shell (sidebar + main)
│   │   └── page.tsx            # Main chat interface
│   ├── books/
│   │   ├── page.tsx            # Book search / browse
│   │   └── [id]/page.tsx       # Book detail page
│   ├── shelves/
│   │   └── page.tsx            # User's reading shelves
│   └── api/
│       └── chat/
│           └── route.ts        # BFF: validates JWT, proxies to agent, streams SSE
├── components/
│   ├── chat/
│   │   ├── ChatWindow.tsx      # Message list + input
│   │   ├── MessageBubble.tsx   # Individual message (user or AI)
│   │   ├── StreamingMessage.tsx # Renders streaming token-by-token
│   │   └── ChatInput.tsx       # Textarea + send button
│   ├── books/
│   │   ├── BookCard.tsx        # Thumbnail, title, author, rating
│   │   ├── BookDetail.tsx      # Full book view
│   │   └── BookSearchBar.tsx   # Debounced search input
│   └── shared/
│       ├── LoadingSkeleton.tsx
│       └── ErrorBoundary.tsx
├── lib/
│   ├── api.ts                  # Typed API client (fetcher wrappers)
│   ├── auth.ts                 # NextAuth config
│   └── stream.ts               # SSE stream reader utilities
└── types/
    ├── book.ts                 # Book, Review, Shelf domain types
    └── chat.ts                 # Message, ChatSession types
```

## Routing Conventions

- `(chat)` is a route group — no URL segment, just shared layout
- `[id]` is a dynamic route — always validate with `zod` before use
- All `page.tsx` files are Server Components by default
- Add `"use client"` only for: event handlers, useState, streaming reads, browser APIs

## API Route Pattern (BFF)

The `api/chat/route.ts` is the **only** entry point to the AI backend from the browser. It must:
1. Validate the JWT from `Authorization: Bearer <token>`
2. Extract `userId` from the token payload
3. Forward to the Agent Orchestrator with `userId` appended
4. Stream the response back using `ReadableStream` + `TransformStream`

Never call the Agent Orchestrator, MCP server, or microservices directly from client components.

## State Management

- **Server state**: SWR for data fetching (books, shelves, user profile)
- **Chat state**: `useReducer` + React Context — not Zustand (keep it simple)
- **No Redux** — overkill for this domain

## Streaming Pattern

```typescript
// app/api/chat/route.ts pattern
export async function POST(req: Request) {
  // 1. Auth
  // 2. Validate body with zod
  // 3. Call agent with fetch + ReadableStream
  // 4. Return TransformStream to client
  // See lib/stream.ts for helpers
}
```

## Styling

- Tailwind CSS v3 only — no custom CSS files except `globals.css`
- Component variants via `class-variance-authority` (cva)
- No inline styles
- Dark mode via `class` strategy (user preference stored in localStorage)

## Performance Rules

- All book images via `next/image` — never raw `<img>`
- Dynamic imports for heavy components: `const BookDetail = dynamic(() => import(...))`
- Chat route must stream — never buffer the full response before sending

## Testing

- `vitest` + `@testing-library/react` for components
- `msw` for API mocking in tests
- No Enzyme — use Testing Library queries only (getByRole, getByText, etc.)
