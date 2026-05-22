// Mock Better-Auth and the auth instance so the route can be imported without DB setup.
jest.mock('@/lib/auth', () => ({ auth: {} }));
jest.mock('better-auth/next-js', () => ({
  toNextJsHandler: () => ({
    GET: jest.fn().mockResolvedValue(new Response('ok', { status: 200 })),
    POST: jest.fn().mockResolvedValue(new Response('ok', { status: 200 })),
  }),
}));

import { NextRequest } from 'next/server';
import { POST } from '../route';

function makePost(pathname: string) {
  return new NextRequest(`http://localhost${pathname}`, { method: 'POST' });
}

describe('POST /api/auth/[...all] — registration guard', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('blocks /sign-up/email with 403 when ALLOW_REGISTRATION=false', async () => {
    process.env.ALLOW_REGISTRATION = 'false';

    const res = await POST(makePost('/api/auth/sign-up/email'), undefined as never);

    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.message).toBe('Registration is disabled');
  });

  it('passes through /sign-up/email when ALLOW_REGISTRATION is unset', async () => {
    delete process.env.ALLOW_REGISTRATION;

    const res = await POST(makePost('/api/auth/sign-up/email'), undefined as never);

    expect(res.status).toBe(200);
  });

  it('passes through /sign-up/email when ALLOW_REGISTRATION=true', async () => {
    process.env.ALLOW_REGISTRATION = 'true';

    const res = await POST(makePost('/api/auth/sign-up/email'), undefined as never);

    expect(res.status).toBe(200);
  });

  it('does not block /sign-in/email regardless of ALLOW_REGISTRATION', async () => {
    process.env.ALLOW_REGISTRATION = 'false';

    const res = await POST(makePost('/api/auth/sign-in/email'), undefined as never);

    expect(res.status).toBe(200);
  });

  it('does not block paths that merely contain sign-up/email as a substring', async () => {
    process.env.ALLOW_REGISTRATION = 'false';

    const res = await POST(makePost('/api/auth/bulk-sign-up/email'), undefined as never);

    expect(res.status).toBe(200);
  });
});
