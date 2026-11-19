import * as React from "react";
import { Camera, RefreshCw, RotateCcw, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface CapturedPhoto {
  blob: Blob;
  url: string;
  width: number;
  height: number;
  filename: string;
}

interface CameraCaptureProps {
  captured: CapturedPhoto | null;
  onCapture: (photo: CapturedPhoto | null) => void;
  bbox?: [number, number, number, number] | null;
  bboxTone?: "verified" | "unknown" | "neutral";
  busy?: boolean;
}

function readImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth || 640, height: img.naturalHeight || 480 });
    img.onerror = () => resolve({ width: 640, height: 480 });
    img.src = url;
  });
}

export async function fileToCapturedPhoto(file: File): Promise<CapturedPhoto> {
  const url = URL.createObjectURL(file);
  const dims = await readImageDimensions(url);
  return {
    blob: file,
    url,
    width: dims.width,
    height: dims.height,
    filename: file.name || "imagen.jpg",
  };
}

export function CameraCapture({
  captured,
  onCapture,
  bbox = null,
  bboxTone = "neutral",
  busy = false,
}: CameraCaptureProps) {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);

  const [cameraActive, setCameraActive] = React.useState(false);
  const [facingMode, setFacingMode] = React.useState<"user" | "environment">("user");
  const [cameraError, setCameraError] = React.useState<string | null>(null);

  const stopStream = React.useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  React.useEffect(() => {
    return () => {
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop();
        }
      }
    };
  }, []);

  const startCamera = React.useCallback(
    async (mode: "user" | "environment" = facingMode) => {
      setCameraError(null);
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError("Este navegador no permite acceso directo a la cámara. Usa 'Subir foto'.");
        return;
      }

      stopStream();
      if (captured) {
        URL.revokeObjectURL(captured.url);
        onCapture(null);
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: mode, width: { ideal: 960 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        setCameraActive(true);
        requestAnimationFrame(() => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            void videoRef.current.play().catch(() => {});
          }
        });
      } catch {
        setCameraError("No se pudo abrir la cámara (permiso denegado o dispositivo no disponible). Puedes subir una foto.");
        setCameraActive(false);
      }
    },
    [captured, facingMode, onCapture, stopStream],
  );

  const handleToggleFacing = () => {
    const next = facingMode === "user" ? "environment" : "user";
    setFacingMode(next);
    if (cameraActive) {
      void startCamera(next);
    }
  };

  const handleSnap = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        stopStream();
        onCapture({
          blob,
          url,
          width: canvas.width,
          height: canvas.height,
          filename: `camara_${Date.now()}.jpg`,
        });
      },
      "image/jpeg",
      0.92,
    );
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setCameraError(null);
    stopStream();
    if (captured) {
      URL.revokeObjectURL(captured.url);
    }

    const photo = await fileToCapturedPhoto(file);
    onCapture(photo);
  };

  const handleReset = () => {
    if (captured) {
      URL.revokeObjectURL(captured.url);
      onCapture(null);
    }
  };

  const bboxStroke =
    bboxTone === "verified"
      ? "#047857"
      : bboxTone === "unknown"
        ? "#b45309"
        : "#0f4c5c";

  return (
    <div className="flex flex-col gap-3">
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-border bg-slate-900">
        {captured ? (
          <>
            <img
              src={captured.url}
              alt="Foto facial lista para procesar"
              className="h-full w-full object-contain bg-slate-950"
            />
            {bbox && bbox[2] > 0 && bbox[3] > 0 && (
              <svg
                viewBox={`0 0 ${captured.width} ${captured.height}`}
                preserveAspectRatio="xMidYMid meet"
                aria-hidden="true"
                className="pointer-events-none absolute inset-0 h-full w-full"
              >
                <rect
                  x={bbox[0]}
                  y={bbox[1]}
                  width={bbox[2]}
                  height={bbox[3]}
                  fill="none"
                  stroke={bboxStroke}
                  strokeWidth={Math.max(3, Math.round(captured.width / 160))}
                  rx={8}
                />
              </svg>
            )}
          </>
        ) : cameraActive ? (
          <>
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className={cn("h-full w-full object-cover", facingMode === "user" && "-scale-x-100")}
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-6 rounded-2xl border border-white/35"
            />
          </>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 text-center text-slate-200">
            <div className="flex h-14 w-14 items-center justify-center rounded-full border border-white/15 bg-white/10">
              <Camera className="h-6 w-6" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-white">Activa la cámara o sube una foto</p>
              <p className="text-xs text-slate-300">
                Encuadra un solo rostro de frente con buena iluminación.
              </p>
            </div>
          </div>
        )}
      </div>

      {cameraError && (
        <p role="alert" className="rounded-lg border border-unknown/30 bg-unknown-soft px-3 py-2 text-xs text-unknown">
          {cameraError}
        </p>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="user"
        onChange={handleFileChange}
        tabIndex={-1}
        aria-hidden="true"
        className="hidden"
      />

      <div className="flex flex-wrap items-center gap-2">
        {!captured && !cameraActive && (
          <Button type="button" onClick={() => void startCamera()} disabled={busy} className="flex-1 min-w-[140px]">
            <Camera className="h-4 w-4" />
            Activar cámara
          </Button>
        )}

        {cameraActive && !captured && (
          <>
            <Button type="button" onClick={handleSnap} disabled={busy} className="flex-1 min-w-[140px]">
              <Camera className="h-4 w-4" />
              Tomar foto
            </Button>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={handleToggleFacing}
              disabled={busy}
              aria-label="Cambiar entre cámara frontal y trasera"
              title="Cambiar cámara"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button type="button" variant="ghost" onClick={stopStream} disabled={busy}>
              Detener
            </Button>
          </>
        )}

        {captured && (
          <Button type="button" variant="outline" onClick={handleReset} disabled={busy} className="flex-1 min-w-[140px]">
            <RotateCcw className="h-4 w-4" />
            Tomar otra foto
          </Button>
        )}

        <Button
          type="button"
          variant="outline"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy}
          className="flex-1 min-w-[130px]"
        >
          <Upload className="h-4 w-4" />
          Subir foto
        </Button>
      </div>
    </div>
  );
}
