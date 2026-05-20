/**
 * Local FS connector form for the wizard's Configure step.
 *
 * A backend-accessible filesystem path.  Distinct from Git: no auth
 * dropdown, no branch field, no URL placeholder competing with the
 * path semantics.  The path can point to a non-git directory of
 * docs/code — ``GitToolkit._ensure_workdir`` resolves it in-place and
 * a subsequent refresh re-reads the same path so files edited manually
 * appear after re-indexing.
 *
 * Path must be absolute and listed in the backend's
 * ``ALLOWED_LOCAL_PATHS`` env var; the backend rejects anything else
 * with a 400 at scan time.  The helper text mentions both requirements
 * up front so users don't waste a scan on a non-allowed path.
 */

import { Alert, Box, TextField, Typography } from '@mui/material';
import type { LocalFsFormState } from '../types';

interface LocalFsConfigureProps {
  data: LocalFsFormState;
  onChange: (next: LocalFsFormState) => void;
  pathError: string | null;
  disabled?: boolean;
}

export function LocalFsConfigure({
  data,
  onChange,
  pathError,
  disabled,
}: LocalFsConfigureProps) {
  return (
    <Box>
      <Alert severity="info" sx={{ mb: 2 }}>
        Local FS reads files directly from a server-side directory.  Refreshing
        the wiki re-reads the same path, so manual edits to files in that
        directory show up after the next refresh — no auth needed.
      </Alert>

      <TextField
        label="Filesystem path"
        placeholder="/srv/docs/my-project"
        value={data.path}
        onChange={(e) => onChange({ path: e.target.value })}
        required
        fullWidth
        margin="normal"
        error={!!pathError}
        helperText={
          pathError ??
          'Absolute path on the backend host.  Must be listed in ALLOWED_LOCAL_PATHS.'
        }
        disabled={disabled}
        inputProps={{ 'data-testid': 'local-fs-path' }}
      />

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
        Examples: <code>/srv/docs/my-project</code> ·{' '}
        <code>/Users/me/notes</code> · <code>/data/handbook</code>
      </Typography>
    </Box>
  );
}
