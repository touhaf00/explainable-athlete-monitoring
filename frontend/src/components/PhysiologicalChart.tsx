import type { PointerEvent } from "react";
import type { SessionReviewTarget } from "../utils/sessionReview";
import { stateLabels } from "../utils/sessionReview";

type Metric = "heartRate" | "respiratoryRate";

interface Props {
  athlete: SessionReviewTarget;
  metric: Metric;
  activeIndex: number;
  onIndexChange: (index: number) => void;
}

const phaseColors = {
  rest: "#dce9f6",
  effort: "#fbe1db",
  recovery: "#dcefe5",
};

export default function PhysiologicalChart({
  athlete,
  metric,
  activeIndex,
  onIndexChange,
}: Props) {
  const session = athlete.session;
  const samples = session.samples;

  if (!samples.length) {
    return <p>Aucune mesure disponible.</p>;
  }

  const isHeartRate = metric === "heartRate";

  const title = isHeartRate
    ? "Fréquence cardiaque"
    : "Fréquence respiratoire";

  const unit = isHeartRate ? "bpm" : "resp/min";
  const color = isHeartRate ? "#ed8977" : "#548bcb";

  const valid = samples
    .map((sample) => sample[metric])
    .filter((v): v is number =>
      v !== null && Number.isFinite(v)
    );

  const observedMin = valid.length
    ? Math.min(...valid)
    : 0;

  const observedMax = valid.length
    ? Math.max(...valid)
    : 100;

  const step = Math.max(
    5,
    Math.ceil(
      (observedMax - observedMin) / 25
    ) * 5
  );

  const min = Math.max(
    0,
    Math.floor(observedMin / step) * step
  );

  const max = Math.max(
    min + step,
    Math.ceil(observedMax / step) * step
  );

  const left = 48;
  const right = 590;
  const top = 24;
  const bottom = 220;

  const width = right - left;
  const height = bottom - top;

  const x = (minute: number) =>
    left + minute / session.duration * width;

  const y = (value: number) =>
    bottom - (value - min) / (max - min) * height;

  const ticks: number[] = [];

  for (let value = min; value <= max; value += step) {
    ticks.push(value);
  }

  const timeTicks = Array.from(
    { length: 6 },
    (_, i) => session.duration * i / 5
  );

  const index = Math.max(
    0,
    Math.min(activeIndex, samples.length - 1)
  );

  const current = samples[index];
  const currentValue = current[metric];

  const phase = session.phases.find(
    (p) =>
      current.minute >= p.startMinute &&
      current.minute < p.endMinute
  );

  let path = "";
  let connected = false;

  for (const sample of samples) {
    const value = sample[metric];

    if (value === null || !Number.isFinite(value)) {
      connected = false;
      continue;
    }

    path += `${connected ? "L" : "M"} ${x(sample.minute)} ${y(value)} `;
    connected = true;
  }

  function inspectPoint(
    event: PointerEvent<SVGRectElement>
  ) {
    const bounds = event.currentTarget
      .getBoundingClientRect();

    if (!bounds.width) return;

    const ratio = Math.max(
      0,
      Math.min(
        1,
        (event.clientX - bounds.left) / bounds.width
      )
    );

    const minute = ratio * session.duration;

    let closest = 0;

    samples.forEach((sample, i) => {
      if (
        Math.abs(sample.minute - minute) <
        Math.abs(samples[closest].minute - minute)
      ) {
        closest = i;
      }
    });

    onIndexChange(closest);
  }

  return (
    <section className="physio-chart">
      <div className="physio-heading">
        <div>
          <h3>{title}</h3>
          <small>
            Mesures de la séance
          </small>
        </div>

        <div className="physio-value" style={{ color }}>
          {currentValue === null ? "—" : currentValue}
          {" "}<small>{unit}</small>
        </div>
      </div>

      <div className="physio-reading">
        <span>Minute {current.minute.toFixed(2)}</span>
        <span>
          {phase
            ? stateLabels[phase.state]
            : "Non annoté"}
        </span>
        <span>
          {session.aggregationMethod === "median"
            ? `Médianes sur ${session.aggregationSeconds} s`
            : "Mesures importées"}
        </span>
      </div>

      <svg
        viewBox="0 0 620 265"
        className="physio-svg"
        role="img"
        aria-label={`${title} de ${athlete.name}`}
      >
        <rect
          x={left}
          y={top}
          width={width}
          height={height}
          fill="#f1f4f7"
        />

        {session.phases.map((phase, i) => (
          <rect
            key={i}
            x={x(phase.startMinute)}
            y={top}
            width={
              x(phase.endMinute) -
              x(phase.startMinute)
            }
            height={height}
            fill={phaseColors[phase.state]}
            opacity=".85"
          />
        ))}

        {ticks.map((value) => (
          <g key={value}>
            <line
              x1={left}
              x2={right}
              y1={y(value)}
              y2={y(value)}
              stroke="#c6cfda"
              strokeDasharray="4 5"
            />
            <text
              x={left - 8}
              y={y(value) + 4}
              textAnchor="end"
              className="physio-axis"
            >
              {value}
            </text>
          </g>
        ))}

        {timeTicks.map((minute, i) => (
          <g key={i}>
            <line
              x1={x(minute)}
              x2={x(minute)}
              y1={top}
              y2={bottom}
              stroke="#c6cfda"
              strokeDasharray="3 6"
            />
            <text
              x={x(minute)}
              y={bottom + 20}
              textAnchor="middle"
              className="physio-axis"
            >
              {minute.toFixed(1)}
            </text>
          </g>
        ))}

        <text
          x={left}
          y={14}
          className="physio-axis"
        >
          {unit}
        </text>

        <text
          x={right}
          y={258}
          textAnchor="end"
          className="physio-axis"
        >
          Temps (min)
        </text>

        <path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        <line
          x1={x(current.minute)}
          x2={x(current.minute)}
          y1={top}
          y2={bottom}
          stroke="#64748b"
          strokeDasharray="4 4"
        />

        {currentValue !== null && (
          <circle
            cx={x(current.minute)}
            cy={y(currentValue)}
            r="5"
            fill={color}
            stroke="white"
            strokeWidth="2"
          />
        )}

        <rect
          x={left}
          y={top}
          width={width}
          height={height}
          fill="transparent"
          onPointerMove={inspectPoint}
          onPointerDown={inspectPoint}
          className="physio-interaction"
        />
      </svg>

      <div className="physio-legend">
        {Object.entries(phaseColors).map(
          ([state, background]) => (
            <span key={state}>
              <i style={{ background }} />
              {stateLabels[
                state as keyof typeof stateLabels
              ]}
            </span>
          )
        )}
        <span>
          <i style={{ background: "#f1f4f7" }} />
          Non annoté
        </span>
      </div>

      <input
        className="physio-slider"
        type="range"
        min={0}
        max={samples.length - 1}
        step={1}
        value={index}
        aria-label={`Explorer ${title.toLowerCase()}`}
        onChange={(event) =>
          onIndexChange(Number(event.target.value))
        }
        style={{ accentColor: color }}
      />

      <small className="physio-help">
        Les interruptions de courbe indiquent des
        valeurs manquantes. Les deux graphiques
        sont synchronisés.
      </small>
    </section>
  );
}
