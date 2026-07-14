"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export type InfoTooltipContent = {
  title: string;
  definition: string;
  formula: string;
  reading: string;
};

const VIEWPORT_MARGIN = 8;
const FALLBACK_PANEL_WIDTH = 280;
const FALLBACK_PANEL_HEIGHT = 140;

/**
 * A small "i" badge rendered inline after a metric label. Opens its
 * definition card on hover, on keyboard focus, and on click/tap (so it
 * isn't mouse-only), and renders that card through a portal into
 * document.body so no ancestor's overflow/rounded card can clip it.
 */
export function InfoTooltip({ content }: { content: InfoTooltipContent }) {
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [clicked, setClicked] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);

  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const panelId = useId();

  const visible = hovered || focused || clicked;

  // Position the portal-rendered panel in viewport coordinates, flipping
  // above the trigger and clamping horizontally so it never runs off-screen.
  useLayoutEffect(() => {
    if (!visible) return;

    function updatePosition() {
      const trigger = triggerRef.current;
      if (!trigger) return;

      const triggerRect = trigger.getBoundingClientRect();
      const panelWidth = panelRef.current?.offsetWidth ?? FALLBACK_PANEL_WIDTH;
      const panelHeight = panelRef.current?.offsetHeight ?? FALLBACK_PANEL_HEIGHT;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      const spaceBelow = viewportHeight - triggerRect.bottom;
      const shouldFlipAbove = spaceBelow < panelHeight + VIEWPORT_MARGIN && triggerRect.top > panelHeight + VIEWPORT_MARGIN;
      const top = shouldFlipAbove
        ? triggerRect.top - panelHeight - VIEWPORT_MARGIN
        : triggerRect.bottom + VIEWPORT_MARGIN;

      let left = triggerRect.left + triggerRect.width / 2 - panelWidth / 2;
      left = Math.max(VIEWPORT_MARGIN, Math.min(left, viewportWidth - panelWidth - VIEWPORT_MARGIN));

      setCoords({ top, left });
    }

    updatePosition();
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);
    return () => {
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [visible]);

  // Tap/click elsewhere dismisses a click-pinned tooltip (mobile has no
  // hover/blur to fall back on).
  useEffect(() => {
    if (!clicked) return;
    function handleDocumentPointerDown(event: MouseEvent) {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || panelRef.current?.contains(target)) return;
      setClicked(false);
    }
    document.addEventListener("mousedown", handleDocumentPointerDown);
    return () => document.removeEventListener("mousedown", handleDocumentPointerDown);
  }, [clicked]);

  useEffect(() => {
    if (!visible) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setHovered(false);
        setFocused(false);
        setClicked(false);
        triggerRef.current?.blur();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [visible]);

  return (
    <span className="relative inline-flex">
      <button
        ref={triggerRef}
        type="button"
        aria-label={`What is ${content.title}?`}
        aria-describedby={visible ? panelId : undefined}
        aria-expanded={visible}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onClick={() => setClicked((prev) => !prev)}
        className="ml-1.5 inline-flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border border-slate-300 text-[9px] italic leading-none text-slate-400 hover:border-slate-400 hover:text-slate-500 dark:border-neutral-600 dark:text-neutral-500 dark:hover:border-neutral-500 dark:hover:text-neutral-400"
      >
        i
      </button>
      {visible &&
        createPortal(
          <div
            ref={panelRef}
            id={panelId}
            role="tooltip"
            style={{
              position: "fixed",
              top: coords?.top ?? -9999,
              left: coords?.left ?? -9999,
            }}
            className="z-50 w-72 max-w-[calc(100vw-1rem)] rounded-xl border border-slate-200 bg-white p-4 text-xs shadow-lg dark:border-neutral-800 dark:bg-neutral-900"
          >
            <p className="font-semibold text-slate-900 dark:text-neutral-50">{content.title}</p>
            <p className="mt-1.5 text-slate-600 dark:text-neutral-400">{content.definition}</p>
            <p className="mt-2 rounded-md bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-800 dark:bg-neutral-800 dark:text-neutral-200">
              {content.formula}
            </p>
            <p className="mt-2 text-slate-600 dark:text-neutral-400">{content.reading}</p>
          </div>,
          document.body,
        )}
    </span>
  );
}
