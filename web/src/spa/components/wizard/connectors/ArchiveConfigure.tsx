/**
 * Archive connector form for the wizard's Configure step.
 *
 * Single-file upload (``.zip`` / ``.tar.gz`` / ``.tgz`` / ``.tar``) that
 * is POSTed immediately to ``/api/v1/uploads/archive``.  On success the
 * backend returns an opaque ``upload_id``; we stash it in
 * ``archive.uploadId`` and the wizard echoes it back as
 * ``scope.archive_upload_id`` on subsequent scan + generate calls.
 *
 * Why upload synchronously here rather than on Submit:
 *
 * - Lets Step 2's Next button gate honestly on extraction success.
 * - Lets the Scan step run against the real extracted directory.
 * - Surfaces traversal / oversize / bad-format errors before the user
 *   advances, instead of failing after they've finished the wizard.
 */

import { useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Stack,
  Typography,
} from '@mui/material';
import { uploadArchive } from '../../../api/wiki';
import type { ArchiveFormState } from '../types';

interface ArchiveConfigureProps {
  data: ArchiveFormState;
  onChange: (next: ArchiveFormState) => void;
  disabled?: boolean;
}

const ACCEPT_ATTR = '.zip,.tar.gz,.tgz,.tar';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export function ArchiveConfigure({ data, onChange, disabled }: ArchiveConfigureProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setUploading(true);
    try {
      const res = await uploadArchive(file);
      onChange({
        uploadId: res.upload_id,
        fileName: file.name,
        fileCount: res.file_count,
        sizeBytes: res.size_bytes,
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      // Leave any prior successful upload in place — the user may have
      // re-picked an invalid file but the previous extraction is still
      // usable. The Alert tells them the new pick failed.
    } finally {
      setUploading(false);
      // Reset the input so picking the same file again retriggers
      // onChange (HTMLInputElement doesn't fire change on identical pick).
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const onPick = () => inputRef.current?.click();

  const hasUpload = data.uploadId !== null;

  return (
    <Box>
      <Alert severity="info" sx={{ mb: 2 }}>
        Upload a <code>.zip</code> or <code>.tar.gz</code> archive of the
        source you want to document. The server extracts it once and reads
        files from the extracted directory — no auth, no branch.
      </Alert>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        style={{ display: 'none' }}
        data-testid="archive-file-input"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />

      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2 }}>
        <Button
          variant="outlined"
          onClick={onPick}
          disabled={disabled || uploading}
          data-testid="archive-pick-file"
        >
          {hasUpload ? 'Replace file' : 'Choose archive'}
        </Button>
        {uploading && <CircularProgress size={20} data-testid="archive-uploading" />}
        {uploading && (
          <Typography variant="body2" color="text.secondary">
            Extracting…
          </Typography>
        )}
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} data-testid="archive-error">
          {error}
        </Alert>
      )}

      {hasUpload && !uploading && (
        <Box
          sx={{
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            p: 1.5,
          }}
          data-testid="archive-summary"
        >
          <Stack spacing={0.5}>
            <Typography variant="body2">
              <strong>File:</strong> {data.fileName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {data.fileCount} files extracted ·{' '}
              {data.sizeBytes !== null ? formatBytes(data.sizeBytes) : ''}
            </Typography>
          </Stack>
        </Box>
      )}
    </Box>
  );
}
