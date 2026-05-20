/**
 * Brand icons for connector cards on Step 1 of the AddSource wizard.
 *
 * Inline SVGs (not @mui/icons-material) so each connector renders with
 * its real brand mark — important for visual recognition on the Pick
 * a Connector grid.
 *
 * Sizing: pass ``size`` (in px) — applied to both ``width`` and
 * ``height``.  Default 28 to match the prior MUI icon sizes.
 *
 * Colour: every brand icon hard-codes its official colour palette.
 * Confluence uses a blue gradient; Git, Jira, and Azure DevOps use
 * single-tone brand colours.  Only ``NotionBrandIcon`` and
 * ``LocalFsBrandIcon`` respect the parent's ``color`` (they paint with
 * ``currentColor``) because they aren't brand-locked.  This is
 * intentional — connector icons are recognised by their brand colour
 * at a glance.
 *
 * The "Git" mark is intentionally NOT GitHub-specific — the Git
 * connector accepts repos from GitHub, GitLab, Bitbucket, and local
 * paths, so the generic Git logo is the honest icon.
 */
import type { SVGProps } from 'react';

const DEFAULT_SIZE = 28;

interface BrandIconProps extends Omit<SVGProps<SVGSVGElement>, 'fill'> {
  size?: number;
}

function svgAttrs(props: BrandIconProps): SVGProps<SVGSVGElement> {
  const { size = DEFAULT_SIZE, ...rest } = props;
  return {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    'aria-hidden': true,
    role: 'img',
    ...rest,
  };
}

// Git — generic logo (red diamond + branches).  Source-shape inspired
// by the official Git logo from https://git-scm.com/downloads/logos.
export function GitBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)}>
      <path
        fill="#F05133"
        d="M23.546 10.93 13.067.452a1.55 1.55 0 0 0-2.193 0L8.694 2.633l2.766 2.766a1.84 1.84 0 0 1 2.331 2.346l2.665 2.665a1.84 1.84 0 0 1 1.904 3.045 1.84 1.84 0 0 1-3.045-1.904l-2.487-2.487v6.547a1.84 1.84 0 1 1-1.51-.039V9.022a1.84 1.84 0 0 1-1-2.413L6.6 3.875.452 10.022a1.55 1.55 0 0 0 0 2.194l10.48 10.48a1.55 1.55 0 0 0 2.193 0l10.42-10.42a1.55 1.55 0 0 0 0-2.193"
      />
    </svg>
  );
}

// Confluence — Atlassian Confluence cloud mark (blue gradient).
export function ConfluenceBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)}>
      <defs>
        <linearGradient id="confluence-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#0052CC" />
          <stop offset="100%" stopColor="#2684FF" />
        </linearGradient>
      </defs>
      <path
        fill="url(#confluence-grad)"
        d="M.87 18.257c-.246.4-.522.86-.756 1.227a.764.764 0 0 0 .258 1.054l4.917 3.027a.764.764 0 0 0 1.057-.253c.196-.328.45-.752.726-1.194 1.945-3.21 3.9-2.84 7.422-1.158l4.881 2.32a.764.764 0 0 0 1.02-.36l2.339-5.29a.764.764 0 0 0-.385-1.003c-1.03-.484-3.078-1.45-4.92-2.342-6.628-3.213-12.264-3.003-16.559 3.972ZM23.13 5.744c.246-.4.523-.86.757-1.228A.764.764 0 0 0 23.63 3.46L18.713.434a.764.764 0 0 0-1.057.253c-.196.328-.45.752-.726 1.194-1.945 3.21-3.9 2.84-7.422 1.158L4.642.72a.764.764 0 0 0-1.02.36L1.282 6.37a.764.764 0 0 0 .385 1.004c1.03.484 3.078 1.45 4.92 2.342 6.643 3.212 12.279 2.997 16.544-3.972Z"
      />
    </svg>
  );
}

// Jira — Atlassian Jira mark (single-tone Jira blue).  Path data sourced
// from Simple Icons (CC0) so the cards/diamonds geometry is correct,
// instead of the hand-derived approximation that previously rendered
// as a misshapen wedge.
export function JiraBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)}>
      <path
        fill="#2684FF"
        d="M11.571 11.513H0a5.218 5.218 0 0 0 5.232 5.215h2.13v2.057A5.215 5.215 0 0 0 12.575 24V12.518a1.005 1.005 0 0 0-1.005-1.005zM17.317 5.756H5.736a5.215 5.215 0 0 0 5.215 5.214h2.129v2.058a5.215 5.215 0 0 0 5.214 5.214V6.76a1.005 1.005 0 0 0-1.005-1.005zM23.063 0H11.483a5.215 5.215 0 0 0 5.214 5.214h2.129v2.058A5.215 5.215 0 0 0 24 12.486V1.005A1.005 1.005 0 0 0 23.063 0z"
      />
    </svg>
  );
}

// Azure DevOps — Microsoft Azure DevOps mark.  Path data sourced from
// Simple Icons (CC0); single-tone Azure blue.
export function AzureDevOpsBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)}>
      <path
        fill="#0078D7"
        d="M0 8.877L2.247 5.91l8.405-3.416V.022l7.37 5.393L2.966 8.338v9.665L0 17.107zm24-4.5v15.124l-5.74 4.5-9.282-3.028v3.005L2.966 17.106l15.275 1.84V5.502z"
      />
    </svg>
  );
}

// Local FS — folder mark.  A plain outlined folder reads as "filesystem
// path" cleanly without competing with the brand marks above.
export function LocalFsBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2v11Z" />
    </svg>
  );
}

// Archive — an outlined box/inventory mark. Renders with currentColor so
// the connector card's ``iconColor`` controls the tint (mirrors the
// Local FS treatment — Archive isn't a brand, it's a content type).
export function ArchiveBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="4" rx="1" />
      <path d="M5 7v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7" />
      <path d="M10 12h4" />
    </svg>
  );
}

// Notion — black "N" logo on transparent background.
export function NotionBrandIcon(props: BrandIconProps) {
  return (
    <svg {...svgAttrs(props)}>
      <path
        fill="currentColor"
        d="M4.459 4.208c.746.606 1.026.56 2.428.466l13.215-.793c.28 0 .047-.28-.046-.326L17.86 1.968c-.42-.326-.981-.7-2.055-.607L3.01 2.295c-.466.046-.56.28-.374.466l1.823 1.447zm.793 3.08v13.904c0 .747.373 1.027 1.214.98l14.523-.84c.841-.046.935-.56.935-1.167V6.354c0-.606-.233-.933-.748-.887l-15.177.887c-.56.047-.747.327-.747.933zm14.337.745c.093.42 0 .84-.42.888l-.7.14v10.264c-.608.327-1.168.514-1.635.514-.748 0-.935-.234-1.495-.933l-4.577-7.186v6.952L12.21 19s0 .84-1.168.84l-3.222.186c-.093-.186 0-.653.327-.746l.84-.233V9.854L7.822 9.76c-.094-.42.14-1.026.793-1.073l3.456-.233 4.764 7.279v-6.44l-1.215-.14c-.093-.514.28-.887.747-.933l3.222-.187z"
      />
    </svg>
  );
}
