import { useEffect, useRef, useState } from 'react';
import { Alert, Box, Dialog, IconButton, Tooltip, Typography } from '@mui/material';
import mermaid from 'mermaid';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import CloseIcon from '@mui/icons-material/Close';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';

interface MermaidDiagramProps {
  chart: string;
  mode?: 'light' | 'dark';
}

let mermaidInitialized = false;

const PAN_STEP = 80;
const SCALE_STEP = 0.15;
const MIN_SCALE = 0.25;
const MAX_SCALE = 10;
const INLINE_MIN_SCALE = 0.25;
const INLINE_MAX_SCALE = 10;

export function MermaidDiagram({ chart, mode = 'dark' }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [naturalWidth, setNaturalWidth] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Inline zoom
  const [scale, setScale] = useState(1);

  // Hover state for controls
  const [hovered, setHovered] = useState(false);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalScale, setModalScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Drag state (refs to avoid re-renders during drag)
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: mode === 'dark' ? 'dark' : 'default',
      securityLevel: 'strict',
      fontFamily: '"Inter", Roboto, Arial, sans-serif',
      fontSize: 14,
      logLevel: 'fatal',
    });
    mermaidInitialized = true;
  }, [mode]);

  useEffect(() => {
    if (!chart || !mermaidInitialized) return;

    const render = async () => {
      try {
        setError(null);
        await mermaid.parse(chart);
        const id = `mermaid-${Math.random().toString(36).slice(2, 11)}`;
        const { svg: rendered } = await mermaid.render(id, chart);
        // Extract natural width from Mermaid's inline style before stripping it.
        const widthMatch = rendered.match(/max-width:\s*([0-9.]+)px/);
        setNaturalWidth(widthMatch ? parseFloat(widthMatch[1]) : null);
        // Remove Mermaid's inline max-width so CSS width can control scaling.
        setSvg(rendered.replace(/max-width:[^;]+;?\s*/g, ''));
      } catch (err) {
        setSvg(null);
        setError(err instanceof Error ? err.message : 'Failed to render diagram');
      }
    };

    render();
  }, [chart, mode]);

  const openModal = () => {
    setModalScale(1);
    setPan({ x: 0, y: 0 });
    setModalOpen(true);
  };

  const closeModal = () => setModalOpen(false);

  // Inline zoom handlers
  const handleInlineZoomIn = (e: React.MouseEvent) => {
    e.stopPropagation();
    setScale((s) => Math.min(INLINE_MAX_SCALE, parseFloat((s + SCALE_STEP).toFixed(2))));
  };

  const handleInlineZoomOut = (e: React.MouseEvent) => {
    e.stopPropagation();
    setScale((s) => Math.max(INLINE_MIN_SCALE, parseFloat((s - SCALE_STEP).toFixed(2))));
  };

  const handleOpenFullscreen = (e: React.MouseEvent) => {
    e.stopPropagation();
    openModal();
  };

  // Modal zoom handlers
  const handleModalZoomIn = () =>
    setModalScale((s) => Math.min(MAX_SCALE, parseFloat((s + SCALE_STEP).toFixed(2))));

  const handleModalZoomOut = () =>
    setModalScale((s) => Math.max(MIN_SCALE, parseFloat((s - SCALE_STEP).toFixed(2))));

  const handleModalReset = () => {
    setModalScale(1);
    setPan({ x: 0, y: 0 });
  };

  // Pan handlers
  const handlePanUp = () => setPan((p) => ({ ...p, y: p.y + PAN_STEP }));
  const handlePanDown = () => setPan((p) => ({ ...p, y: p.y - PAN_STEP }));
  const handlePanLeft = () => setPan((p) => ({ ...p, x: p.x + PAN_STEP }));
  const handlePanRight = () => setPan((p) => ({ ...p, x: p.x - PAN_STEP }));

  // Drag-to-pan in modal
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    e.preventDefault();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  // Scroll-to-zoom in modal
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setModalScale((s) =>
      Math.max(MIN_SCALE, Math.min(MAX_SCALE, parseFloat((s + (e.deltaY < 0 ? SCALE_STEP : -SCALE_STEP)).toFixed(2))))
    );
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          Failed to render Mermaid diagram
        </Typography>
        <Typography
          variant="body2"
          component="pre"
          sx={{ fontFamily: 'monospace', fontSize: '11px', whiteSpace: 'pre-wrap', m: 0, mt: 0.5 }}
        >
          {error}
        </Typography>
      </Alert>
    );
  }

  if (!svg) return null;

  const pillBg = 'rgba(0,0,0,0.55)';
  const pillColor = '#fff';

  return (
    <>
      {/* Diagram card with hover controls */}
      <Box
        ref={containerRef}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={openModal}
        sx={{
          position: 'relative',
          my: 3,
          p: 2,
          borderRadius: 2,
          bgcolor: mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
          border: '1px solid',
          borderColor: 'divider',
          overflow: 'auto',
          cursor: 'zoom-in',
          // Use width (layout-affecting) instead of zoom/transform:scale so
          // the card expands correctly in all browsers including Firefox.
          '& .mermaid-svg-wrap svg': {
            display: 'block',
            width: naturalWidth ? `${naturalWidth * scale}px` : '100%',
            height: 'auto',
          },
        }}
      >
        {/* SVG — zoom applied via .mermaid-svg-wrap selector above */}
        <Box
          className="mermaid-svg-wrap"
          sx={{ lineHeight: 0 }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />

        {/* Floating hover controls pill */}
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 0.25,
            bgcolor: pillBg,
            borderRadius: '20px',
            px: 0.5,
            py: 0.25,
            opacity: hovered ? 1 : 0,
            transition: 'opacity 200ms ease',
            pointerEvents: hovered ? 'auto' : 'none',
          }}
        >
          <Tooltip title="Zoom out" placement="bottom">
            <IconButton
              size="small"
              aria-label="Zoom out"
              onClick={handleInlineZoomOut}
              sx={{ color: pillColor, p: 0.5 }}
              disabled={scale <= INLINE_MIN_SCALE}
            >
              <ZoomOutIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Zoom in" placement="bottom">
            <IconButton
              size="small"
              aria-label="Zoom in"
              onClick={handleInlineZoomIn}
              sx={{ color: pillColor, p: 0.5 }}
              disabled={scale >= INLINE_MAX_SCALE}
            >
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Fullscreen" placement="bottom">
            <IconButton
              size="small"
              aria-label="Open fullscreen"
              onClick={handleOpenFullscreen}
              sx={{ color: pillColor, p: 0.5 }}
            >
              <FullscreenIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Reset zoom" placement="bottom">
            <IconButton
              size="small"
              aria-label="Reset zoom"
              onClick={(e) => { e.stopPropagation(); setScale(1); }}
              sx={{ color: pillColor, p: 0.5 }}
              disabled={scale === 1}
            >
              <FitScreenIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Fullscreen modal */}
      <Dialog
        open={modalOpen}
        onClose={closeModal}
        fullScreen
        PaperProps={{
          sx: {
            bgcolor: mode === 'dark' ? '#0d1117' : '#f6f8fa',
            overflow: 'hidden',
            position: 'relative',
          },
        }}
      >
        {/* Close button */}
        <Box sx={{ position: 'absolute', top: 12, right: 12, zIndex: 10 }}>
          <Tooltip title="Close">
            <IconButton
              aria-label="Close fullscreen"
              onClick={closeModal}
              sx={{
                bgcolor: pillBg,
                color: '#fff',
                '&:hover': { bgcolor: 'rgba(0,0,0,0.75)' },
              }}
            >
              <CloseIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Zoom % indicator */}
        <Box
          sx={{
            position: 'absolute',
            top: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            bgcolor: pillBg,
            color: '#fff',
            px: 1.5,
            py: 0.5,
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: 500,
            letterSpacing: '0.02em',
            zIndex: 10,
            userSelect: 'none',
          }}
        >
          {Math.round(modalScale * 100)}%
        </Box>

        {/* Pannable / zoomable SVG canvas — fills the whole dialog */}
        <Box
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          sx={{
            position: 'absolute',
            inset: 0,
            overflow: 'hidden',
            cursor: isDragging.current ? 'grabbing' : 'grab',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Box
            sx={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${modalScale})`,
              transformOrigin: 'center center',
              transition: isDragging.current ? 'none' : 'transform 50ms ease',
              lineHeight: 0,
              '& svg': {
                display: 'block',
              },
            }}
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        </Box>

        {/* GitHub-style control pad — bottom-center */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 10,
            bgcolor: pillBg,
            borderRadius: 2,
            p: 0.5,
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 36px)',
            gridTemplateRows: 'repeat(3, 36px)',
            gap: 0.25,
            alignItems: 'center',
            justifyItems: 'center',
          }}
        >
          {/* Row 1: [empty] [up] [zoom+] */}
          <Box />
          <Tooltip title="Pan up">
            <IconButton aria-label="Pan up" size="small" onClick={handlePanUp} sx={{ color: pillColor, p: 0.5 }}>
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom in">
            <IconButton
              aria-label="Zoom in"
              size="small"
              onClick={handleModalZoomIn}
              sx={{ color: pillColor, p: 0.5 }}
              disabled={modalScale >= MAX_SCALE}
            >
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* Row 2: [left] [reset] [right] */}
          <Tooltip title="Pan left">
            <IconButton aria-label="Pan left" size="small" onClick={handlePanLeft} sx={{ color: pillColor, p: 0.5 }}>
              <ArrowBackIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset view">
            <IconButton aria-label="Reset view" size="small" onClick={handleModalReset} sx={{ color: pillColor, p: 0.5 }}>
              <FitScreenIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Pan right">
            <IconButton aria-label="Pan right" size="small" onClick={handlePanRight} sx={{ color: pillColor, p: 0.5 }}>
              <ArrowForwardIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* Row 3: [empty] [down] [zoom-] */}
          <Box />
          <Tooltip title="Pan down">
            <IconButton aria-label="Pan down" size="small" onClick={handlePanDown} sx={{ color: pillColor, p: 0.5 }}>
              <ArrowDownwardIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom out">
            <IconButton
              aria-label="Zoom out"
              size="small"
              onClick={handleModalZoomOut}
              sx={{ color: pillColor, p: 0.5 }}
              disabled={modalScale <= MIN_SCALE}
            >
              <ZoomOutIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Dialog>
    </>
  );
}
