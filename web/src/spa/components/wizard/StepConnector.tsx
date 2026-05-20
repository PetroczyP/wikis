/**
 * Step 1 of the AddSource wizard: pick a connector.
 *
 * Cartograph-style 6-card 3-column grid. Four connectors are live (Git,
 * Confluence, Jira, Local FS); two are placeholders (Azure DevOps,
 * Notion) marked "Soon" — disabled and dimmed so the surface communicates
 * the roadmap without offering a broken choice. Clicking a live card both
 * sets ``source_type`` on the wizard state and advances to Step 2.
 */

import {
  Box,
  Card,
  CardActionArea,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  ArchiveBrandIcon,
  AzureDevOpsBrandIcon,
  ConfluenceBrandIcon,
  GitBrandIcon,
  JiraBrandIcon,
  LocalFsBrandIcon,
  NotionBrandIcon,
} from './connectors/BrandIcons';
import type { WikiSourceType } from '../../api/wiki';

// ---------------------------------------------------------------------------
// Connector catalogue (discriminated union — TS guarantees that only
// `disabled: false` entries carry a real WikiSourceType id, so flipping a
// placeholder's `disabled` flag without also widening the API shape is a
// compile error.)
// ---------------------------------------------------------------------------

type PlaceholderId = 'ado' | 'notion';

// Per-card unique key separate from the runtime ``id``.  Two cards can
// share an ``id`` (Local FS routes through ``source_type='git'`` like
// the Git card) — the ``key`` field stays unique so React list keys,
// active-state highlighting, and the click handler all agree on which
// card the user picked.
type ConnectorKey =
  | 'git'
  | 'confluence'
  | 'jira'
  | 'ado'
  | 'local_fs'
  | 'archive'
  | 'notion';

type ConnectorOption =
  | {
      key: ConnectorKey;
      disabled: false;
      id: WikiSourceType;
      label: string;
      description: string;
      icon: React.ReactNode;
      iconColor: string;
      testId: string;
    }
  | {
      key: ConnectorKey;
      disabled: true;
      id: PlaceholderId;
      label: string;
      description: string;
      icon: React.ReactNode;
      iconColor: string;
      testId: string;
    };

const CONNECTORS: ConnectorOption[] = [
  {
    key: 'git',
    disabled: false,
    id: 'git',
    label: 'Git',
    description: 'GitHub, GitLab, Bitbucket, Azure DevOps',
    icon: <GitBrandIcon />,
    iconColor: 'text.primary',
    testId: 'connector-git',
  },
  {
    key: 'confluence',
    disabled: false,
    id: 'confluence',
    label: 'Confluence',
    description: 'Atlassian Cloud spaces',
    icon: <ConfluenceBrandIcon />,
    iconColor: 'text.primary',
    testId: 'connector-confluence',
  },
  {
    key: 'jira',
    disabled: false,
    id: 'jira',
    label: 'Jira',
    description: 'Issues matching a JQL query',
    icon: <JiraBrandIcon />,
    iconColor: 'text.primary',
    testId: 'connector-jira',
  },
  {
    key: 'ado',
    disabled: true,
    id: 'ado',
    label: 'Azure DevOps',
    description: 'Repos & boards',
    icon: <AzureDevOpsBrandIcon />,
    iconColor: 'text.primary',
    testId: 'connector-ado',
  },
  {
    // Local FS — backend-accessible filesystem path. Routes through the
    // Git connector because GitToolkit.test_connection already supports
    // "Connected to local directory (non-git)" — the wizard just lands
    // the user on the same configure form with a path-friendly placeholder.
    // ``key`` stays unique even though ``id === 'git'`` so React keys and
    // the per-card active state don't collide with the Git card.
    key: 'local_fs',
    disabled: false,
    id: 'git',
    label: 'Local FS',
    description: 'Filesystem path on the server',
    icon: <LocalFsBrandIcon />,
    iconColor: '#FB923C',
    testId: 'connector-local-fs',
  },
  {
    // Archive — uploads a .zip/.tar.gz and routes through the Local FS
    // pipeline once the backend extracts it.  Shares ``id: 'git'`` with
    // the Git and Local FS cards because the on-wire ``source_type`` is
    // the same; the per-card ``key`` keeps active-state highlighting
    // honest when Git/Local FS/Archive coexist on the picker.
    key: 'archive',
    disabled: false,
    id: 'git',
    label: 'Archive',
    description: 'Upload a zip or tar.gz of the source',
    icon: <ArchiveBrandIcon />,
    iconColor: '#A78BFA',
    testId: 'connector-archive',
  },
  {
    key: 'notion',
    disabled: true,
    id: 'notion',
    label: 'Notion',
    description: 'Databases & pages',
    icon: <NotionBrandIcon />,
    iconColor: 'text.primary',
    testId: 'connector-notion',
  },
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface StepConnectorProps {
  selected: WikiSourceType;
  /**
   * ``selectedLocalFs`` distinguishes the Local FS card from the Git card
   * when ``selected === 'git'``.  Both cards share ``id: 'git'`` (the
   * backend doesn't have a distinct local-fs source type yet), so the
   * wizard tracks the choice separately on ``WizardFormData.localFsMode``
   * and forwards it here for the active-state computation.
   */
  selectedLocalFs: boolean;
  /**
   * ``selectedArchive`` mirrors ``selectedLocalFs`` for the Archive card
   * (#archive). Archive, Local FS, and Git all wire to ``id: 'git'`` on
   * the backend; the wizard distinguishes them on the picker via these
   * flags so the active-state border ends up on the right card.
   */
  selectedArchive: boolean;
  onSelect: (source: WikiSourceType, opts: { localFs: boolean; archive: boolean }) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function StepConnector({ selected, selectedLocalFs, selectedArchive, onSelect }: StepConnectorProps) {
  return (
    <Stack spacing={2} sx={{ mt: 0.5 }}>
      <Typography variant="body2" color="text.secondary">
        Choose the type of data source you want to connect.
      </Typography>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)' },
          gap: 1.25,
        }}
      >
        {CONNECTORS.map((c) => {
          // ``id`` can be shared (Local FS and Archive both reuse
          // ``id: 'git'``).  Use the per-card ``key`` together with the
          // mode flags so only the card the user actually clicked
          // highlights on revisit.
          const isLocalFsCard = c.key === 'local_fs';
          const isArchiveCard = c.key === 'archive';
          let activeForVariant: boolean;
          if (isLocalFsCard) {
            activeForVariant = selectedLocalFs;
          } else if (isArchiveCard) {
            activeForVariant = selectedArchive;
          } else {
            // Plain Git card — active only when neither alt-mode flag is set.
            activeForVariant = !selectedLocalFs && !selectedArchive;
          }
          const active = !c.disabled && c.id === selected && activeForVariant;
          const card = (
            <Card
              variant="outlined"
              sx={{
                borderRadius: 2,
                borderColor: active ? 'primary.main' : 'divider',
                borderWidth: active ? 2 : 1,
                opacity: c.disabled ? 0.4 : 1,
                position: 'relative',
                transition: 'border-color 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease',
                '&:hover': c.disabled
                  ? {}
                  : {
                      borderColor: 'primary.main',
                      boxShadow: '0 0 14px rgba(168, 85, 247, 0.22)',
                    },
              }}
            >
              <CardActionArea
                onClick={() => {
                  if (!c.disabled)
                    onSelect(c.id, {
                      localFs: c.key === 'local_fs',
                      archive: c.key === 'archive',
                    });
                }}
                disabled={c.disabled}
                data-testid={c.testId}
                sx={{ p: 1.75, minHeight: 120 }}
              >
                <Stack alignItems="center" spacing={0.75}>
                  <Box sx={{ color: c.iconColor, display: 'flex' }}>{c.icon}</Box>
                  <Typography
                    component="span"
                    sx={{ fontSize: '0.875rem', fontWeight: 600, lineHeight: 1.2 }}
                  >
                    {c.label}
                  </Typography>
                  <Typography
                    component="span"
                    sx={{
                      fontSize: '0.6875rem',
                      color: 'text.secondary',
                      lineHeight: 1.3,
                      textAlign: 'center',
                      minHeight: 24,
                    }}
                  >
                    {c.description}
                  </Typography>
                </Stack>

                {c.disabled && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 6,
                      right: 6,
                      fontSize: '0.625rem',
                      fontWeight: 700,
                      letterSpacing: '0.05em',
                      textTransform: 'uppercase',
                      color: 'text.secondary',
                      bgcolor: 'background.paper',
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 0.75,
                      px: 0.6,
                      py: 0.15,
                    }}
                  >
                    Soon
                  </Box>
                )}
              </CardActionArea>
            </Card>
          );

          return (
            <Box key={c.key}>
              {c.disabled ? (
                <Tooltip arrow placement="top" title="Coming soon">
                  <Box>{card}</Box>
                </Tooltip>
              ) : (
                card
              )}
            </Box>
          );
        })}
      </Box>
    </Stack>
  );
}
