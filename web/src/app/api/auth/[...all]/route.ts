import { auth } from '@/lib/auth';
import { toNextJsHandler } from 'better-auth/next-js';
import { NextRequest, NextResponse } from 'next/server';

const handlers = toNextJsHandler(auth);

export const GET = handlers.GET;

// Next.js requires the context param for catch-all routes; Better-Auth routes by URL so we don't forward it.
export async function POST(request: NextRequest, _context: { params: Promise<{ all: string[] }> }) {
  const url = new URL(request.url);
  if (process.env.ALLOW_REGISTRATION === 'false' && url.pathname === '/api/auth/sign-up/email') {
    return NextResponse.json({ message: 'Registration is disabled' }, { status: 403 });
  }
  return handlers.POST(request);
}
