const { useState, useEffect, useRef } = React;

function ProgressPage() {
  // 1. State Management
  const [weights, setWeights] = useState({
    start: "",
    week1: "",
    week2: "",
    week3: "",
    week4: "",
    week5: "",
    week6: "",
  });

  const [plateauDetected, setPlateauDetected] = useState(false);
  const [newPlanGenerated, setNewPlanGenerated] = useState(false);

  // goalMode: 'loss' | 'gain' | 'maintain'
  const [goalMode, setGoalMode] = useState("loss");

  // simple save status
  const [saveStatus, setSaveStatus] = useState("idle"); // "idle" | "saving" | "saved" | "error"

  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  const labels = [
    "Start",
    "Week 1",
    "Week 2",
    "Week 3",
    "Week 4",
    "Week 5",
    "Week 6",
  ];

  const orderedKeys = [
    "start",
    "week1",
    "week2",
    "week3",
    "week4",
    "week5",
    "week6",
  ];

  const handleWeightChange = (key, value) => {
    setWeights((prev) => ({
      ...prev,
      [key]: value,
    }));
    // once user edits, reset "saved" indicator
    if (saveStatus === "saved") {
      setSaveStatus("idle");
    }
  };

  const parse = (v) => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
  };

  // === NEW: Load progress from backend on mount ===
  useEffect(() => {
    async function fetchProgress() {
      try {
        const res = await fetch("/api/progress/weights", {
          method: "GET",
          headers: {
            "Accept": "application/json",
          },
        });
        if (!res.ok) {
          console.warn("Failed to fetch progress:", res.status);
          return;
        }
        const data = await res.json();

        if (data.weights) {
          const incoming = data.weights;
          setWeights((prev) => {
            const next = { ...prev };
            orderedKeys.forEach((key) => {
              const val = incoming[key];
              next[key] = val == null ? "" : String(val);
            });
            return next;
          });
        }

        if (data.goal_mode) {
          setGoalMode(data.goal_mode);
        }

        if (data.plateau && typeof data.plateau.detected === "boolean") {
          setPlateauDetected(data.plateau.detected);
          setNewPlanGenerated(data.plateau.detected);
        }
      } catch (err) {
        console.error("Error loading progress", err);
      }
    }

    fetchProgress();
  }, []);

  // === NEW: Save progress to backend ===
  const handleSave = async () => {
    setSaveStatus("saving");
    try {
      const payloadWeights = {};
      orderedKeys.forEach((key) => {
        const n = parse(weights[key]);
        payloadWeights[key] = Number.isFinite(n) ? n : null;
      });

      const res = await fetch("/api/progress/weights", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          weights: payloadWeights,
          goal_mode: goalMode,
        }),
      });

      if (!res.ok) {
        console.error("Save failed with status", res.status);
        setSaveStatus("error");
        return;
      }

      const data = await res.json();

      if (data.weights) {
        const incoming = data.weights;
        setWeights((prev) => {
          const next = { ...prev };
          orderedKeys.forEach((key) => {
            const val = incoming[key];
            next[key] = val == null ? "" : String(val);
          });
          return next;
        });
      }

      if (data.plateau && typeof data.plateau.detected === "boolean") {
        setPlateauDetected(data.plateau.detected);
        setNewPlanGenerated(data.plateau.detected);
      }

      setSaveStatus("saved");
    } catch (err) {
      console.error("Error saving progress", err);
      setSaveStatus("error");
    }
  };

  // 3. Build chart data (Actual + Predicted)
  const buildChartData = () => {
    const actualData = orderedKeys.map((k) => parse(weights[k]));

    const startWeight = parse(weights.start);
    const predictedData = orderedKeys.map((_, idx) => {
      if (startWeight == null) return null;
      if (idx === 0) return startWeight;

      // Adjust slope based on goal mode
      if (goalMode === "gain") {
        // gentle gain, e.g. +0.25 kg / week
        return startWeight + 0.25 * idx;
      }
      if (goalMode === "maintain") {
        // flat line around start weight
        return startWeight;
      }
      // default: fat loss, -0.5 kg per week
      return startWeight - 0.5 * idx;
    });

    return { actualData, predictedData };
  };

  // 4. Plateau Logic (trend-based, safer) – client-side
  useEffect(() => {
    const numericValues = orderedKeys.map((k) => parse(weights[k]));
    const startWeight = numericValues[0];

    const filledWeeks = [];
    for (let i = 1; i < numericValues.length; i += 1) {
      if (numericValues[i] != null) {
        filledWeeks.push({ idx: i, value: numericValues[i] });
      }
    }

    if (!startWeight || filledWeeks.length < 3) {
      setPlateauDetected(false);
      setNewPlanGenerated(false);
      return;
    }

    const latest = filledWeeks[filledWeeks.length - 1];

    const netChange = Math.abs(startWeight - latest.value);
    const negligibleNetChange = netChange < 0.5;

    let hasMeaningfulDrop = false;
    const DROP_THRESHOLD = 0.2;

    for (let i = 1; i < filledWeeks.length; i += 1) {
      const prev = filledWeeks[i - 1].value;
      const curr = filledWeeks[i].value;
      if (prev != null && curr != null && prev - curr >= DROP_THRESHOLD) {
        hasMeaningfulDrop = true;
        break;
      }
    }

    const plateau = negligibleNetChange || !hasMeaningfulDrop;

    if (plateau) {
      setPlateauDetected(true);
      setNewPlanGenerated(true);
    } else {
      setPlateauDetected(false);
      setNewPlanGenerated(false);
    }
  }, [weights]);

  // 5. Graph logic with Chart.js
  useEffect(() => {
    if (!chartRef.current || typeof Chart === "undefined") return;

    const { actualData, predictedData } = buildChartData();

    const data = {
      labels,
      datasets: [
        {
          label: "Actual Weight",
          data: actualData,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.18)",
          borderWidth: 3,
          tension: 0.35,
          pointRadius: 4,
          pointHoverRadius: 6,
          spanGaps: true,
        },
        {
          label:
            goalMode === "gain"
              ? "Predicted (+0.25kg/week)"
              : goalMode === "maintain"
              ? "Predicted (maintenance)"
              : "Predicted (−0.5kg/week)",
          data: predictedData,
          borderColor: "#9ca3af",
          backgroundColor: "transparent",
          borderWidth: 2,
          borderDash: [6, 6],
          tension: 0.3,
          pointRadius: 0,
          spanGaps: true,
        },
      ],
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: "index", intersect: false },
      },
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          grid: { color: "rgba(148, 163, 184, 0.18)" },
          ticks: { color: "#e5e7eb" },
        },
        y: {
          grid: { color: "rgba(148, 163, 184, 0.18)" },
          ticks: {
            color: "#e5e7eb",
            callback: (value) => `${value} kg`,
          },
        },
      },
    };

    if (chartInstanceRef.current) {
      chartInstanceRef.current.data = data;
      chartInstanceRef.current.options = options;
      chartInstanceRef.current.update();
    } else {
      chartInstanceRef.current = new Chart(chartRef.current, {
        type: "line",
        data,
        options,
      });
    }

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [weights, goalMode]);

  const renderInput = (label, key, helper) => (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        padding: "12px 14px",
        borderRadius: "12px",
        background: "rgba(17, 24, 39, 0.9)",
        border: "1px solid rgba(55, 65, 81, 0.9)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <span
          style={{
            fontSize: "0.9rem",
            fontWeight: 600,
            color: "#e5e7eb",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: "0.7rem",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            padding: "2px 8px",
            borderRadius: "999px",
            background:
              key === "start"
                ? "rgba(59, 130, 246, 0.2)"
                : "rgba(16, 185, 129, 0.16)",
            color: key === "start" ? "#bfdbfe" : "#6ee7b7",
            border:
              key === "start"
                ? "1px solid rgba(59, 130, 246, 0.6)"
                : "1px solid rgba(16, 185, 129, 0.55)",
          }}
        >
          {key === "start" ? "baseline" : key.replace("week", "week ")}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <input
          type="number"
          step="0.1"
          value={weights[key]}
          onChange={(e) => handleWeightChange(key, e.target.value)}
          placeholder={helper || "e.g. 66.4"}
          style={{
            flex: 1,
            padding: "8px 10px",
            borderRadius: "8px",
            border: "1px solid rgba(75, 85, 99, 0.9)",
            background: "rgba(15, 23, 42, 0.9)",
            color: "#f9fafb",
            fontSize: "0.85rem",
          }}
        />
        <span
          style={{
            fontSize: "0.8rem",
            color: "#9ca3af",
          }}
        >
          kg
        </span>
      </div>
    </div>
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: "18px",
      }}
    >
      {/* 2. Input section title */}
      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        <h2
          style={{
            fontSize: "1.2rem",
            fontWeight: 700,
            color: "#f9fafb",
            letterSpacing: "0.02em",
          }}
        >
          Log Your Weekly Weight
        </h2>
        <p
          style={{
            fontSize: "0.85rem",
            color: "#9ca3af",
          }}
        >
          Keep your check-ins up to date. We’ll compare your actual curve with your
          target and trigger metabolic adjustments when progress stalls.
        </p>
      </div>

      {/* Input grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: "10px",
        }}
      >
        {renderInput("Starting Weight", "start", "e.g. 66.0 (baseline)")}
        {renderInput("Week 1", "week1", "e.g. 65.6")}
        {renderInput("Week 2", "week2", "e.g. 65.1")}
        {renderInput("Week 3", "week3", "e.g. 64.7")}
        {renderInput("Week 4", "week4", "e.g. 64.3")}
        {renderInput("Week 5", "week5", "e.g. 63.9")}
        {renderInput("Week 6", "week6", "e.g. 63.4")}
      </div>

      {/* small chips under inputs */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "8px",
          marginTop: "2px",
        }}
      >
        <span
          style={{
            fontSize: "0.75rem",
            padding: "4px 10px",
            borderRadius: "999px",
            background: "rgba(16, 185, 129, 0.18)",
            color: "#6ee7b7",
            border: "1px solid rgba(16, 185, 129, 0.6)",
          }}
        >
          Live-linked to progress graph
        </span>
        <span
          style={{
            fontSize: "0.75rem",
            padding: "4px 10px",
            borderRadius: "999px",
            background: "rgba(59, 130, 246, 0.18)",
            color: "#bfdbfe",
            border: "1px solid rgba(59, 130, 246, 0.65)",
          }}
        >
          Plateau detection active
        </span>
        {/* Save status indicator */}
        {saveStatus === "saved" && (
          <span
            style={{
              fontSize: "0.75rem",
              padding: "4px 10px",
              borderRadius: "999px",
              background: "rgba(16, 185, 129, 0.18)",
              color: "#bbf7d0",
              border: "1px solid rgba(16, 185, 129, 0.6)",
            }}
          >
            Progress saved
          </span>
        )}
        {saveStatus === "error" && (
          <span
            style={{
              fontSize: "0.75rem",
              padding: "4px 10px",
              borderRadius: "999px",
              background: "rgba(248, 113, 113, 0.18)",
              color: "#fee2e2",
              border: "1px solid rgba(248, 113, 113, 0.7)",
            }}
          >
            Error saving progress
          </span>
        )}
      </div>

      {/* Prediction mode + Save button */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "8px",
          marginTop: "4px",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            alignItems: "center",
          }}
        >
          <span
            style={{
              fontSize: "0.8rem",
              color: "#9ca3af",
            }}
          >
            Prediction mode:
          </span>
          {["loss", "gain", "maintain"].map((mode) => {
            const active = goalMode === mode;
            const labels = {
              loss: "Fat loss",
              gain: "Muscle gain",
              maintain: "Maintain weight",
            };
            return (
              <button
                key={mode}
                type="button"
                onClick={() => setGoalMode(mode)}
                style={{
                  borderRadius: "999px",
                  padding: "4px 10px",
                  fontSize: "0.78rem",
                  cursor: "pointer",
                  border: active
                    ? "1px solid rgba(16, 185, 129, 0.9)"
                    : "1px solid rgba(55, 65, 81, 0.9)",
                  background: active
                    ? "rgba(16, 185, 129, 0.18)"
                    : "rgba(15, 23, 42, 0.9)",
                  color: active ? "#6ee7b7" : "#d1d5db",
                }}
              >
                {labels[mode]}
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={handleSave}
          disabled={saveStatus === "saving"}
          style={{
            padding: "6px 14px",
            borderRadius: "999px",
            border: "1px solid rgba(34, 197, 94, 0.85)",
            background:
              saveStatus === "saving"
                ? "rgba(34, 197, 94, 0.18)"
                : "linear-gradient(135deg, rgba(34,197,94,0.9), rgba(22,163,74,0.9))",
            color: "#ecfdf5",
            fontSize: "0.8rem",
            fontWeight: 600,
            cursor: saveStatus === "saving" ? "default" : "pointer",
          }}
        >
          {saveStatus === "saving" ? "Saving..." : "Save Progress"}
        </button>
      </div>

      {/* Graph card */}
      <div
        style={{
          marginTop: "6px",
          padding: "14px 14px 12px 14px",
          borderRadius: "14px",
          background:
            "radial-gradient(circle at top left, rgba(52, 211, 153, 0.16), transparent 55%), rgba(15,23,42,0.97)",
          border: "1px solid rgba(55, 65, 81, 0.95)",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          flex: "1 1 auto",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            <span
              style={{
                fontSize: "0.95rem",
                fontWeight: 600,
                color: "#e5e7eb",
              }}
            >
              Weight Progress Curve
            </span>
            <span
              style={{
                fontSize: "0.78rem",
                color: "#9ca3af",
              }}
            >
              Actual check-ins vs. your predicted trajectory.
            </span>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              fontSize: "0.78rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span
                style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "999px",
                  background: "#22c55e",
                }}
              />
              <span style={{ color: "#d1d5db" }}>Actual</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span
                style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "999px",
                  border: "2px dashed #9ca3af",
                  background: "transparent",
                }}
              />
              <span style={{ color: "#9ca3af" }}>Predicted</span>
            </div>
          </div>
        </div>

        <div
          style={{
            position: "relative",
            width: "100%",
            height: "230px",
          }}
        >
          <canvas ref={chartRef} />
        </div>
      </div>

      {/* 4. Plateau alert & New plan card */}
      {plateauDetected && (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div
            style={{
              borderRadius: "14px",
              padding: "10px 12px",
              background:
                "linear-gradient(135deg, rgba(248, 113, 113, 0.16), rgba(252, 165, 165, 0.08))",
              border: "1px solid rgba(248, 113, 113, 0.8)",
              display: "flex",
              alignItems: "flex-start",
              gap: "10px",
            }}
          >
            <div
              style={{
                width: "26px",
                height: "26px",
                borderRadius: "999px",
                background: "rgba(248, 113, 113, 0.18)",
                border: "1px solid rgba(248, 113, 113, 0.75)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "1.1rem",
              }}
            >
              ⚠️
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span
                style={{
                  fontSize: "0.95rem",
                  fontWeight: 700,
                  color: "#fee2e2",
                }}
              >
                ⚠️ Stagnation Detected. AI has switched your plan to Metabolic
                Boost Mode.
              </span>
              <p
                style={{
                  fontSize: "0.82rem",
                  color: "#fecaca",
                }}
              >
                Your most recent check-ins show virtually no change week to week. To
                protect your motivation and keep fat loss moving, we’ve activated a more
                aggressive, but still sustainable, phase of your program.
              </p>
            </div>
          </div>

          {newPlanGenerated && (
            <div
              style={{
                borderRadius: "14px",
                padding: "12px 14px",
                background:
                  "linear-gradient(130deg, rgba(16, 185, 129, 0.16), rgba(59, 130, 246, 0.12))",
                border: "1px solid rgba(55, 65, 81, 0.95)",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: "10px",
                }}
              >
                <div
                  style={{ display: "flex", flexDirection: "column", gap: "2px" }}
                >
                  <span
                    style={{
                      fontSize: "0.95rem",
                      fontWeight: 600,
                      color: "#ecfdf5",
                    }}
                  >
                    New Plan: Metabolic Boost
                  </span>
                  <span
                    style={{
                      fontSize: "0.8rem",
                      color: "#d1fae5",
                    }}
                  >
                    Tweaks below are applied conceptually to your current NutriFit
                    plan.
                  </span>
                </div>
                <span
                  style={{
                    fontSize: "0.75rem",
                    padding: "4px 9px",
                    borderRadius: "999px",
                    background: "rgba(15, 23, 42, 0.78)",
                    color: "#a5b4fc",
                    border: "1px solid rgba(79, 70, 229, 0.85)",
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                  }}
                >
                  AI Optimized
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                  gap: "8px",
                  marginTop: "4px",
                }}
              >
                <div
                  style={{
                    padding: "8px 9px",
                    borderRadius: "10px",
                    background: "rgba(15, 23, 42, 0.96)",
                    border: "1px solid rgba(21, 128, 61, 0.85)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#bbf7d0",
                      fontWeight: 600,
                    }}
                  >
                    Calories reduced by 200
                  </span>
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#dcfce7",
                    }}
                  >
                    Slight cut targeting refined carbs and late-night snacking
                    windows.
                  </span>
                </div>

                <div
                  style={{
                    padding: "8px 9px",
                    borderRadius: "10px",
                    background: "rgba(15, 23, 42, 0.96)",
                    border: "1px solid rgba(59, 130, 246, 0.9)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#bfdbfe",
                      fontWeight: 600,
                    }}
                  >
                    Added HIIT session
                  </span>
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#dbeafe",
                    }}
                  >
                    2× weekly 12–15 minute HIIT finishers after strength days.
                  </span>
                </div>

                <div
                  style={{
                    padding: "8px 9px",
                    borderRadius: "10px",
                    background: "rgba(15, 23, 42, 0.96)",
                    border: "1px solid rgba(234, 179, 8, 0.9)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#facc15",
                      fontWeight: 600,
                    }}
                  >
                    Recovery protocol
                  </span>
                  <span
                    style={{
                      fontSize: "0.78rem",
                      color: "#fef9c3",
                    }}
                  >
                    Priority on 7–8h sleep, hydration and step target to keep
                    hormones aligned with fat loss.
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Expose component globally so progress.html can mount it
window.ProgressPage = ProgressPage;