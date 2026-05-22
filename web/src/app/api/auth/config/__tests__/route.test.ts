import { GET } from '../route';

describe('GET /api/auth/config', () => {
  const WATCHED = [
    'GITHUB_CLIENT_ID',
    'GITHUB_CLIENT_SECRET',
    'GOOGLE_CLIENT_ID',
    'GOOGLE_CLIENT_SECRET',
    'ALLOW_REGISTRATION',
  ] as const;
  const saved: Partial<Record<string, string>> = {};

  beforeEach(() => {
    for (const key of WATCHED) saved[key] = process.env[key];
  });

  afterEach(() => {
    for (const key of WATCHED) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
  });

  it('returns github=true when both GitHub secrets are set', async () => {
    process.env.GITHUB_CLIENT_ID = 'id';
    process.env.GITHUB_CLIENT_SECRET = 'secret';
    delete process.env.GOOGLE_CLIENT_ID;
    delete process.env.GOOGLE_CLIENT_SECRET;

    const res = await GET();
    const body = await res.json();

    expect(body.github).toBe(true);
    expect(body.google).toBe(false);
  });

  it('returns github=false when only client ID is set (no secret)', async () => {
    process.env.GITHUB_CLIENT_ID = 'id';
    delete process.env.GITHUB_CLIENT_SECRET;

    const res = await GET();
    const body = await res.json();

    expect(body.github).toBe(false);
  });

  it('returns google=true when both Google secrets are set', async () => {
    delete process.env.GITHUB_CLIENT_ID;
    process.env.GOOGLE_CLIENT_ID = 'gid';
    process.env.GOOGLE_CLIENT_SECRET = 'gsecret';

    const res = await GET();
    const body = await res.json();

    expect(body.google).toBe(true);
    expect(body.github).toBe(false);
  });

  it('registrationEnabled defaults to true when ALLOW_REGISTRATION is unset', async () => {
    delete process.env.ALLOW_REGISTRATION;

    const res = await GET();
    const body = await res.json();

    expect(body.registrationEnabled).toBe(true);
  });

  it('registrationEnabled is false when ALLOW_REGISTRATION=false', async () => {
    process.env.ALLOW_REGISTRATION = 'false';

    const res = await GET();
    const body = await res.json();

    expect(body.registrationEnabled).toBe(false);
  });

  it('registrationEnabled is true when ALLOW_REGISTRATION=true', async () => {
    process.env.ALLOW_REGISTRATION = 'true';

    const res = await GET();
    const body = await res.json();

    expect(body.registrationEnabled).toBe(true);
  });

  it('includes Cache-Control header', async () => {
    const res = await GET();
    expect(res.headers.get('Cache-Control')).toBe('public, max-age=300, stale-while-revalidate=60');
  });
});
