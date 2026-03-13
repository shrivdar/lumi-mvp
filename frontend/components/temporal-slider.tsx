"use client";

import { useState, useMemo } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAnimationTimer } from "@/lib/hooks";

interface TemporalSliderProps {
  timestamps: string[]; // ISO timestamps of all KG mutations
  onChange: (range: [string, string]) => void;
  className?: string;
}

export default function TemporalSlider({ timestamps, onChange, className }: TemporalSliderProps) {
  const sorted = useMemo(
    () => [...timestamps].sort((a, b) => new Date(a).getTime() - new Date(b).getTime()),
    [timestamps],
  );

  const [position, setPosition] = useState(100); // 0-100 percent
  const [playing, setPlaying] = useState(false);

  const tick = useAnimationTimer(playing ? 100 : 999999);

  // Auto-advance when playing
  useMemo(() => {
    if (playing && position < 100) {
      const next = Math.min(100, position + 0.5);
      setPosition(next);
      emitRange(next);
    } else if (playing && position >= 100) {
      setPlaying(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  function emitRange(pct: number) {
    if (sorted.length === 0) return;
    const idx = Math.floor((pct / 100) * (sorted.length - 1));
    onChange([sorted[0], sorted[Math.min(idx, sorted.length - 1)]]);
  }

  function handleSlider(e: React.ChangeEvent<HTMLInputElement>) {
    const val = +e.target.value;
    setPosition(val);
    emitRange(val);
  }

  if (sorted.length < 2) return null;

  return (
    <div className={cn("flex items-center gap-3 rounded-lg bg-gray-900 px-3 py-2", className)}>
      <button
        onClick={() => { setPosition(0); emitRange(0); }}
        className="text-gray-500 hover:text-gray-300"
      >
        <SkipBack className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => setPlaying(!playing)}
        className="text-gray-500 hover:text-gray-300"
      >
        {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
      </button>
      <button
        onClick={() => { setPosition(100); emitRange(100); }}
        className="text-gray-500 hover:text-gray-300"
      >
        <SkipForward className="h-3.5 w-3.5" />
      </button>

      <input
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={position}
        onChange={handleSlider}
        className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-gray-700 accent-pathway"
      />

      <span className="w-10 text-right text-[10px] tabular-nums text-gray-500">
        {Math.round(position)}%
      </span>
    </div>
  );
}
