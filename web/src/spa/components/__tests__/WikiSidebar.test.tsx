/**
 * @jest-environment jsdom
 */
import '@testing-library/jest-dom';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { WikiSidebar } from '../WikiSidebar';
import type { WikiPage } from '../WikiSidebar';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePages(overrides: Partial<WikiPage>[] = []): WikiPage[] {
  return overrides.map((o, i) => ({
    id: o.id ?? `page-${i}`,
    title: o.title ?? `Page ${i}`,
    order: o.order ?? i,
    section: o.section,
  }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('WikiSidebar — flat pages (no section)', () => {
  it('renders all pages as flat list when no section is set', () => {
    const pages = makePages([
      { id: 'intro', title: 'Introduction' },
      { id: 'api', title: 'API Reference' },
    ]);
    render(<WikiSidebar pages={pages} activePageId={null} onSelectPage={jest.fn()} />);

    expect(screen.getByText('Introduction')).toBeInTheDocument();
    expect(screen.getByText('API Reference')).toBeInTheDocument();
  });

  it('calls onSelectPage with the page id when a page is clicked', () => {
    const onSelect = jest.fn();
    const pages = makePages([{ id: 'intro', title: 'Introduction' }]);
    render(<WikiSidebar pages={pages} activePageId={null} onSelectPage={onSelect} />);

    fireEvent.click(screen.getByText('Introduction'));
    expect(onSelect).toHaveBeenCalledWith('intro');
  });
});

describe('WikiSidebar — chaptered pages (#287)', () => {
  const chapteredPages: WikiPage[] = [
    // Chapter index (_index)
    {
      id: 'workers-chapter/_index',
      title: 'Workers Chapter',
      order: 0,
      section: 'Workers Chapter',
    },
    // Sub-pages under 'Workers Chapter'
    {
      id: 'workers-chapter/worker-lifecycle',
      title: 'Worker Lifecycle',
      order: 1,
      section: 'Workers Chapter',
    },
    {
      id: 'workers-chapter/test-launch-rest-api',
      title: 'Test Launch REST API',
      order: 2,
      section: 'Workers Chapter',
    },
    // Flat page without a chapter
    {
      id: 'overview',
      title: 'Overview',
      order: 3,
      section: undefined,
    },
  ];

  it('groups sub-pages under their chapter section', () => {
    render(
      <WikiSidebar pages={chapteredPages} activePageId={null} onSelectPage={jest.fn()} />
    );

    expect(screen.getByText('Worker Lifecycle')).toBeInTheDocument();
    expect(screen.getByText('Test Launch REST API')).toBeInTheDocument();
    // Section header text is present — CSS textTransform: uppercase doesn't change DOM text.
    expect(screen.getByText('Workers Chapter')).toBeInTheDocument();
  });

  it('renders pages without a section as flat items', () => {
    render(
      <WikiSidebar pages={chapteredPages} activePageId={null} onSelectPage={jest.fn()} />
    );
    expect(screen.getByText('Overview')).toBeInTheDocument();
  });

  it('treats _index page as the section header, not a list item in the collapse', () => {
    const onSelect = jest.fn();
    render(
      <WikiSidebar pages={chapteredPages} activePageId={null} onSelectPage={onSelect} />
    );

    // Click the section header Typography text — it should navigate to the _index page.
    // CSS textTransform: uppercase doesn't change the DOM text node.
    fireEvent.click(screen.getByText('Workers Chapter'));
    expect(onSelect).toHaveBeenCalledWith('workers-chapter/_index');
  });

  it('collapses and expands chapter sub-pages on toggle button click', () => {
    render(
      <WikiSidebar pages={chapteredPages} activePageId={null} onSelectPage={jest.fn()} />
    );

    // Initially open — toggle arrow shows ▾.
    expect(screen.getByText('▾')).toBeInTheDocument();

    // Click the toggle arrow to collapse.
    const toggleArrow = screen.getByText('▾');
    fireEvent.click(toggleArrow);

    // After collapse the toggle arrow switches to ▸ (closed state).
    expect(screen.getByText('▸')).toBeInTheDocument();
    expect(screen.queryByText('▾')).not.toBeInTheDocument();
  });

  it('navigates to sub-page when sub-page is clicked', () => {
    const onSelect = jest.fn();
    render(
      <WikiSidebar pages={chapteredPages} activePageId={null} onSelectPage={onSelect} />
    );

    fireEvent.click(screen.getByText('Worker Lifecycle'));
    expect(onSelect).toHaveBeenCalledWith('workers-chapter/worker-lifecycle');
  });

  it('highlights active sub-page', () => {
    render(
      <WikiSidebar
        pages={chapteredPages}
        activePageId="workers-chapter/test-launch-rest-api"
        onSelectPage={jest.fn()}
      />
    );

    // The active page text should be rendered — presence confirms it's shown.
    expect(screen.getByText('Test Launch REST API')).toBeInTheDocument();
    // The ListItemButton closest parent div should carry the Mui-selected class.
    const activeText = screen.getByText('Test Launch REST API');
    const btn = activeText.closest('[class*="MuiListItemButton"]');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveClass('Mui-selected');
  });
});
