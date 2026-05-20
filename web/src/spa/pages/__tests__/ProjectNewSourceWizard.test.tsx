/**
 * @jest-environment jsdom
 *
 * Project-page "+ New source" wizard tests.
 *
 * The Ingestion tab was removed; new-wiki generation from the project page
 * now opens the AddSourceWizard as a modal triggered by the "+ New source"
 * dashed card on the Overview grid.
 *
 * Covers:
 *  1. Owner: the "+ New source" card opens the wizard modal.
 *  2. Successful generation links the new wiki to the project and navigates
 *     to /wiki/:id with the generating-indicator query params.
 *  3. Link failure surfaces a warning but still navigates.
 *  4. 409 already-exists routes to the existing wiki without calling addWikiToProject.
 *  5. Legacy ``/project/:id/ingestion`` URLs redirect to ``/project/:id`` (no tab).
 *  6. Non-owner: neither dashed card is rendered.
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

// ---------------------------------------------------------------------------
// Mock project API
// ---------------------------------------------------------------------------
const mockAddWikiToProject = jest.fn();
const mockGetProject = jest.fn();
const mockListProjectWikis = jest.fn();

jest.mock('../../api/project', () => ({
  getProject: (...args: unknown[]) => mockGetProject(...args),
  listProjectWikis: (...args: unknown[]) => mockListProjectWikis(...args),
  updateProject: jest.fn(),
  deleteProject: jest.fn(),
  addWikiToProject: (...args: unknown[]) => mockAddWikiToProject(...args),
  removeWikiFromProject: jest.fn(),
}));

// ---------------------------------------------------------------------------
// Mock wiki API (generateWikiMultiSource used by the wizard)
// ---------------------------------------------------------------------------
const mockGenerateWikiMultiSource = jest.fn();
const mockScanSource = jest.fn();

jest.mock('../../api/wiki', () => {
  const actual = jest.requireActual('../../api/wiki');
  return {
    ...actual,
    generateWikiMultiSource: (...args: unknown[]) => mockGenerateWikiMultiSource(...args),
    scanSource: (...args: unknown[]) => mockScanSource(...args),
    listWikis: jest.fn().mockResolvedValue({ wikis: [] }),
  };
});

// ---------------------------------------------------------------------------
// Mock useConnections (used inside AddSourceWizard)
// ---------------------------------------------------------------------------
jest.mock('../../hooks/useConnections', () => ({
  useConnections: () => ({
    connections: [],
    atlassian: null,
    refreshAtlassianIfNeeded: jest.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Mock react-markdown + plugins (not available in jsdom)
// ---------------------------------------------------------------------------
jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));
jest.mock('rehype-highlight', () => ({ __esModule: true, default: () => {} }));

// ---------------------------------------------------------------------------
// Mock mermaid-dependent components (mermaid ships ESM only)
// ---------------------------------------------------------------------------
jest.mock('../../components/MermaidDiagram', () => ({ MermaidDiagram: () => null }));
jest.mock('../../components/AnswerView', () => ({ AnswerView: () => null }));
jest.mock('../../components/AnswerHeader', () => ({ AnswerHeader: () => null }));
jest.mock('../../components/ToolCallPanel', () => ({ ToolCallPanel: () => null }));
jest.mock('../../components/CodeMapTree', () => ({ __esModule: true, default: () => null }));
jest.mock('../../components/RecomputeWidget', () => ({ RecomputeWidget: () => null }));
jest.mock('../../components/AskBar', () => ({ AskBar: () => null }));

// ---------------------------------------------------------------------------
// Capture navigation calls
// ---------------------------------------------------------------------------
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => {
  const actual = jest.requireActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

import { ProjectProvider } from '../../context/ProjectContext';
import { ProjectPage } from '../ProjectPage';

const PROJECT_ID = 'proj-123';

const mockProjectOwner = {
  id: PROJECT_ID,
  name: 'Test Project',
  description: null,
  visibility: 'personal' as const,
  owner_id: 'user-1',
  is_owner: true,
  created_at: '2025-01-01T00:00:00Z',
  wiki_count: 0,
};

const mockProjectNonOwner = {
  ...mockProjectOwner,
  is_owner: false,
};

function setup(initialPath = `/project/${PROJECT_ID}`, project = mockProjectOwner) {
  mockGetProject.mockResolvedValue(project);
  mockListProjectWikis.mockResolvedValue({ wikis: [] });

  render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route
          path="project/:projectId"
          element={
            <ProjectProvider>
              <ProjectPage />
            </ProjectProvider>
          }
        />
        <Route
          path="project/:projectId/:tab"
          element={
            <ProjectProvider>
              <ProjectPage />
            </ProjectProvider>
          }
        />
        <Route path="*" element={<div data-testid="navigated-page" />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  mockScanSource.mockResolvedValue({
    source_type: 'git',
    reachable: true,
    preview: {
      default_branch: null,
      resolved_branch: 'main',
      commit_hash: null,
      file_count: 5,
      top_paths: [],
      size_bytes: 0,
    },
    warnings: [],
  });
});

describe('Project page — "+ New source" wizard', () => {
  it('owner: clicking "+ New source" opens the AddSourceWizard modal', async () => {
    const user = userEvent.setup();
    setup();
    // Wait for the project page to load and the dashed cards to render.
    await screen.findByTestId('project-new-source');

    // Wizard is NOT rendered initially.
    expect(screen.queryByTestId('add-source-wizard')).not.toBeInTheDocument();

    await user.click(screen.getByTestId('project-new-source'));

    // Wizard now mounted.
    expect(await screen.findByTestId('add-source-wizard')).toBeInTheDocument();
    expect(screen.getByText('Connector')).toBeInTheDocument();
  });

  it('success: links the new wiki to this project and navigates with generating params', async () => {
    const user = userEvent.setup();
    mockGenerateWikiMultiSource.mockResolvedValue({
      wiki_id: 'owner--repo--main',
      invocation_id: 'inv-42',
      status: 'generating',
      message: 'ok',
    });
    mockAddWikiToProject.mockResolvedValue({ ...mockProjectOwner, wiki_count: 1 });

    setup();
    await user.click(await screen.findByTestId('project-new-source'));
    await screen.findByTestId('add-source-wizard');

    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() =>
      expect(mockAddWikiToProject).toHaveBeenCalledWith(PROJECT_ID, 'owner--repo--main'),
    );

    const navCall = mockNavigate.mock.calls.find((args: string[]) =>
      (args[0] as string).includes('/wiki/owner--repo--main'),
    );
    expect(navCall).toBeTruthy();
    expect(navCall?.[0]).toContain('generating=true');
    expect(navCall?.[0]).toContain('invocation=inv-42');
  });

  it('link failure shows a warning but still navigates to the new wiki', async () => {
    const user = userEvent.setup();
    mockGenerateWikiMultiSource.mockResolvedValue({
      wiki_id: 'owner--repo--main',
      invocation_id: null,
      status: 'generating',
      message: 'ok',
    });
    mockAddWikiToProject.mockRejectedValue(new Error('Server error'));

    setup();
    await user.click(await screen.findByTestId('project-new-source'));
    await screen.findByTestId('add-source-wizard');

    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        expect.stringContaining('/wiki/owner--repo--main'),
      ),
    );

    expect(
      await screen.findByText(/could not be automatically linked/i),
    ).toBeInTheDocument();
  });

  it('409 already-exists navigates to existing wiki without calling addWikiToProject', async () => {
    const user = userEvent.setup();
    const conflictErr = Object.assign(new Error('conflict'), {
      status: 409,
      body: { detail: { wiki_id: 'existing-wiki-id' } },
    });
    mockGenerateWikiMultiSource.mockRejectedValue(conflictErr);

    setup();
    await user.click(await screen.findByTestId('project-new-source'));
    await screen.findByTestId('add-source-wizard');

    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByTestId('wizard-next'));
    await screen.findByTestId('scan-success');
    await user.click(screen.getByTestId('wizard-next'));
    await user.click(screen.getByTestId('wizard-submit'));

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith('/wiki/existing-wiki-id'),
    );
    expect(mockAddWikiToProject).not.toHaveBeenCalled();
  });

  it('legacy /project/:id/ingestion URL redirects to /project/:id (no tab anymore)', async () => {
    setup(`/project/${PROJECT_ID}/ingestion`, mockProjectOwner);

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(`/project/${PROJECT_ID}`, { replace: true }),
    );
  });

  it('non-owner: neither dashed card is rendered', async () => {
    setup(`/project/${PROJECT_ID}`, mockProjectNonOwner);

    expect(await screen.findByText('Test Project')).toBeInTheDocument();

    // Neither owner-only card is shown.
    expect(screen.queryByTestId('project-new-source')).not.toBeInTheDocument();
    expect(screen.queryByTestId('project-add-repository')).not.toBeInTheDocument();

    // The empty-state alert is visible since this project has no wikis.
    expect(screen.getByText(/no wikis in this project yet/i)).toBeInTheDocument();
  });

  it('closing the wizard fully unmounts it and discards form state on next open', async () => {
    // Regression for Copilot on #280: the wizard used to be rendered with
    // ``open={false}`` while closed, which retained its component-local
    // form state — including a pasted PAT.  Now the component is mounted
    // ONLY while open, so closing fully unmounts and the next open starts
    // fresh.
    const user = userEvent.setup();
    setup();
    await user.click(await screen.findByTestId('project-new-source'));
    await screen.findByTestId('add-source-wizard');

    // Walk a couple of steps into the wizard with a pasted PAT.
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByRole('combobox', { name: /authentication/i }));
    await user.click(await screen.findByText(/Paste token once/i));
    await user.type(screen.getByTestId('pasted-pat-input'), 'ghp_secret_value');

    // Cancel the wizard. The wizard is dirty (PAT was typed) so a
    // "Discard changes?" confirm dialog pops up — confirm it.
    await user.click(screen.getByRole('button', { name: /cancel/i }));
    await user.click(await screen.findByTestId('wizard-discard-confirm-button'));

    // Wizard must be fully gone from the DOM (not just hidden).
    expect(screen.queryByTestId('add-source-wizard')).not.toBeInTheDocument();
    expect(screen.queryByTestId('pasted-pat-input')).not.toBeInTheDocument();

    // Reopen — the wizard should mount fresh.  Walk back into the PAT
    // field and verify the previously-pasted value is GONE.
    await user.click(screen.getByTestId('project-new-source'));
    await screen.findByTestId('add-source-wizard');
    await user.click(screen.getByTestId('connector-git'));
    await user.type(screen.getByTestId('git-repo-url'), 'https://github.com/owner/repo');
    await user.click(screen.getByRole('combobox', { name: /authentication/i }));
    await user.click(await screen.findByText(/Paste token once/i));

    const patField = screen.getByTestId('pasted-pat-input') as HTMLInputElement;
    expect(patField.value).toBe('');
  });

  it('DashboardPage also unmounts the wizard on close (source-level guard)', () => {
    // DashboardPage has no jest test harness of its own. Rather than
    // build one just for this guard, assert the unmount pattern exists
    // in the source — Rio 2nd-pass on #280 noted the dashboard could
    // silently regress to ``open={showGenerateModal}`` (always mounted)
    // without this check.
    const fs = require('fs') as typeof import('fs');
    const path = require('path') as typeof import('path');
    const dashboardSrc = fs.readFileSync(
      path.resolve(__dirname, '../DashboardPage.tsx'),
      'utf-8',
    );
    expect(dashboardSrc).toMatch(/\{showGenerateModal\s*&&\s*\(/);
    // And NOT the always-mounted form that would retain form state.
    expect(dashboardSrc).not.toMatch(/<AddSourceWizard[^>]*open=\{showGenerateModal\}/);
  });
});
