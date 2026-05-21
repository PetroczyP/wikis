/**
 * Git connector form for the wizard's Configure step.
 *
 * Authentication is simplified: a single optional PAT field.
 * Empty = public/anonymous (patSource='none'), filled = PAT auth (patSource='paste').
 * The dropdown selector was removed — the token field presence is the only
 * signal needed.
 */

import { Box, TextField } from '@mui/material';
import type { GitFormState } from '../types';

interface GitConfigureProps {
  data: GitFormState;
  onChange: (next: GitFormState) => void;
  urlError: string | null;
  disabled?: boolean;
}

export function GitConfigure({ data, onChange, urlError, disabled }: GitConfigureProps) {
  const set = <K extends keyof GitFormState>(key: K, value: GitFormState[K]) =>
    onChange({ ...data, [key]: value });

  const handlePatChange = (value: string) => {
    onChange({
      ...data,
      pastedPat: value,
      // Derive patSource from whether the field is filled — no dropdown needed.
      patSource: value.trim() ? 'paste' : 'none',
    });
  };

  return (
    <Box>
      <TextField
        label="Repository URL"
        placeholder="https://github.com/owner/repo"
        value={data.repo_url}
        onChange={(e) => set('repo_url', e.target.value)}
        required
        fullWidth
        margin="normal"
        error={!!urlError}
        helperText={urlError ?? undefined}
        disabled={disabled}
        inputProps={{ 'data-testid': 'git-repo-url' }}
      />

      <TextField
        label="Branch"
        value={data.branch}
        onChange={(e) => set('branch', e.target.value)}
        fullWidth
        margin="normal"
        disabled={disabled}
        inputProps={{ 'data-testid': 'git-branch' }}
      />

      <TextField
        label="Personal Access Token"
        placeholder="Leave empty for public repos"
        value={data.pastedPat}
        onChange={(e) => handlePatChange(e.target.value)}
        fullWidth
        margin="normal"
        type="password"
        disabled={disabled}
        helperText={data.pastedPat.trim() ? 'Token will not be stored' : 'Leave empty for public repositories'}
        inputProps={{ 'data-testid': 'pasted-pat-input' }}
      />
    </Box>
  );
}
