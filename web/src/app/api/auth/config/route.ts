import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json(
    {
      github: !!(process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET),
      google: !!(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET),
      registrationEnabled: process.env.ALLOW_REGISTRATION !== 'false',
    },
    { headers: { 'Cache-Control': 'public, max-age=300, stale-while-revalidate=60' } },
  );
}
