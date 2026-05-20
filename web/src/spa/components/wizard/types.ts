/**
 * Shared state shape for the AddSource wizard (#208).
 *
 * One canonical form-state object lives in the wizard container; each step
 * reads/writes the slice it owns. Prevents prop-drilling 15 setters from
 * the original GenerateForm and makes it trivial to remount the wizard with
 * a partial preset (e.g. URL paste from the dashboard search box).
 */

import type { WikiSourceType } from '../../api/wiki';

export type PatSource = 'none' | 'paste';

export interface GitFormState {
  repo_url: string;
  branch: string;
  patSource: PatSource;
  pastedPat: string;
}

export interface ConfluenceFormState {
  space_keys: string[];
}

export interface JiraFormState {
  jql: string;
}

/**
 * Archive form state — the result of an upload to
 * ``POST /api/v1/uploads/archive``.
 *
 * The Archive connector is "Local FS with a one-time extraction step":
 * the user uploads a ``.zip``/``.tar.gz`` and the backend returns an
 * opaque ``upload_id``. The wizard echoes that id back as
 * ``scope.archive_upload_id`` with ``source_type='git'``; the backend
 * resolves it to the extracted directory server-side so host paths
 * never cross the wire.
 *
 * Slots are nullable so the wizard can detect "no upload yet" without
 * a parallel boolean. ``uploadId`` is the only field ``configureValid``
 * cares about; the others are display-only.
 */
export interface ArchiveFormState {
  uploadId: string | null;
  fileName: string | null;
  fileCount: number | null;
  sizeBytes: number | null;
}

/**
 * Local-FS form state — a backend-accessible filesystem path.
 *
 * Submitted with ``source_type='git'`` (the backend's ``GitToolkit``
 * handles non-git directories in-place via ``_ensure_workdir`` without
 * cloning, so a refresh re-reads the user's path and picks up files
 * they edited manually).  ``LocalFsFormState`` is kept distinct from
 * ``GitFormState`` so the Local FS configure form can omit the auth
 * dropdown and branch field entirely.
 */
export interface LocalFsFormState {
  path: string;
}

/**
 * Atlassian credentials chosen at the Configure step.  The wizard supports
 * two authentication shapes:
 *
 * - ``"oauth"``: uses the in-app OAuth flow; credentials come from
 *   ``useConnections().atlassian`` and are NOT mirrored into this form
 *   state (kept as the single source of truth in the connections store).
 * - ``"api_token"``: HTTP Basic auth using an Atlassian-issued API token
 *   (see https://id.atlassian.com/manage-profile/security/api-tokens).
 *   ``siteUrl``, ``email``, and ``apiToken`` are required.
 */
export type AtlassianAuthMode = 'oauth' | 'api_token';

export interface AtlassianBasicAuthFormState {
  siteUrl: string;
  email: string;
  apiToken: string;
}

export interface WizardFormData {
  source_type: WikiSourceType;
  /**
   * Distinguishes Local FS from regular Git when ``source_type='git'``.
   * The Local FS path is taken from ``localFs.path``; on the wire we still
   * submit ``source_type='git'`` since the backend's ``GitToolkit`` accepts
   * both git URLs and bare filesystem paths through the same code path.
   */
  localFsMode: boolean;
  /**
   * Distinguishes Archive from regular Git when ``source_type='git'``.
   * Mutually exclusive with ``localFsMode`` — picking either of the
   * non-URL Git surfaces clears the other.  The upload token is tracked
   * on ``archive.uploadId`` and submitted as
   * ``scope.archive_upload_id``; the backend resolves it to the
   * extracted directory server-side.
   */
  archiveMode: boolean;
  git: GitFormState;
  localFs: LocalFsFormState;
  archive: ArchiveFormState;
  confluence: ConfluenceFormState;
  jira: JiraFormState;
  atlassianAuthMode: AtlassianAuthMode;
  atlassianBasic: AtlassianBasicAuthFormState;
  wiki_title: string;
}

export const INITIAL_FORM_DATA: WizardFormData = {
  source_type: 'git',
  localFsMode: false,
  archiveMode: false,
  git: {
    repo_url: '',
    branch: 'main',
    patSource: 'none',
    pastedPat: '',
  },
  localFs: { path: '' },
  archive: {
    uploadId: null,
    fileName: null,
    fileCount: null,
    sizeBytes: null,
  },
  confluence: { space_keys: [] },
  jira: { jql: 'ORDER BY created DESC' },
  atlassianAuthMode: 'oauth',
  atlassianBasic: { siteUrl: '', email: '', apiToken: '' },
  wiki_title: '',
};

/**
 * Step index enum mirrored as integers for MUI Stepper.
 *
 * Kept centralised so step components and the container agree without
 * magic numbers floating around.
 */
export const WIZARD_STEPS = ['Connector', 'Configure', 'Scan', 'Confirm'] as const;
export type WizardStepIndex = 0 | 1 | 2 | 3;
