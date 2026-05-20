/**
 * Step 2 of the AddSource wizard: connector-specific configuration.
 *
 * Pure dispatch component — delegates to the appropriate ``*Configure``
 * form.  ``source_type`` discriminates the four major flows; the
 * ``localFsMode`` flag further splits ``source_type='git'`` into the
 * Git form (URL + auth) and the Local FS form (path only, no auth) so
 * the two surfaces don't have to share fields.  Validation errors are
 * computed by the container and forwarded via the per-connector error
 * props so the Next button can be gated centrally.
 */

import { Box } from '@mui/material';
import { ArchiveConfigure } from './connectors/ArchiveConfigure';
import { GitConfigure } from './connectors/GitConfigure';
import { LocalFsConfigure } from './connectors/LocalFsConfigure';
import { ConfluenceConfigure } from './connectors/ConfluenceConfigure';
import { JiraConfigure } from './connectors/JiraConfigure';
import type { WizardFormData } from './types';

interface StepConfigureProps {
  data: WizardFormData;
  onChange: (next: WizardFormData) => void;
  urlError: string | null;
  pathError: string | null;
  spaceKeysError: string | null;
  jqlError: string | null;
  disabled?: boolean;
}

export function StepConfigure({
  data,
  onChange,
  urlError,
  pathError,
  spaceKeysError,
  jqlError,
  disabled,
}: StepConfigureProps) {
  return (
    <Box sx={{ mt: 0.5 }}>
      {data.source_type === 'git' && data.archiveMode && (
        <ArchiveConfigure
          data={data.archive}
          onChange={(archive) => onChange({ ...data, archive })}
          disabled={disabled}
        />
      )}
      {data.source_type === 'git' && data.localFsMode && (
        <LocalFsConfigure
          data={data.localFs}
          onChange={(localFs) => onChange({ ...data, localFs })}
          pathError={pathError}
          disabled={disabled}
        />
      )}
      {data.source_type === 'git' && !data.localFsMode && !data.archiveMode && (
        <GitConfigure
          data={data.git}
          onChange={(git) => onChange({ ...data, git })}
          urlError={urlError}
          disabled={disabled}
        />
      )}
      {data.source_type === 'confluence' && (
        <ConfluenceConfigure
          data={data.confluence}
          onChange={(confluence) => onChange({ ...data, confluence })}
          spaceKeysError={spaceKeysError}
          disabled={disabled}
          authMode={data.atlassianAuthMode}
          onAuthModeChange={(atlassianAuthMode) => onChange({ ...data, atlassianAuthMode })}
          basicAuth={data.atlassianBasic}
          onBasicAuthChange={(atlassianBasic) => onChange({ ...data, atlassianBasic })}
        />
      )}
      {data.source_type === 'jira' && (
        <JiraConfigure
          data={data.jira}
          onChange={(jira) => onChange({ ...data, jira })}
          jqlError={jqlError}
          disabled={disabled}
          authMode={data.atlassianAuthMode}
          onAuthModeChange={(atlassianAuthMode) => onChange({ ...data, atlassianAuthMode })}
          basicAuth={data.atlassianBasic}
          onBasicAuthChange={(atlassianBasic) => onChange({ ...data, atlassianBasic })}
        />
      )}
    </Box>
  );
}
