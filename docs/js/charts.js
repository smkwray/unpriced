/* unpriced — all Chart.js chart initialization */

(function () {
  'use strict';

  var charts = [];

  /* ── Curve generator: constant-elasticity ── */
  function generateCurve(P0, Q0, elasticity, pMin, pMax, nPoints) {
    var pts = [];
    for (var i = 0; i < nPoints; i++) {
      var p = pMin + (pMax - pMin) * i / (nPoints - 1);
      var q = Q0 * Math.pow(p / P0, elasticity);
      pts.push({ x: q, y: p });
    }
    return pts;
  }

  /* ── Number formatting helpers ── */
  function fmtDollar(v) { return '$' + v.toLocaleString('en-US', { maximumFractionDigits: 0 }); }
  function fmtM(v) { return (v / 1e6).toFixed(2) + 'M'; }
  function fmtPct(v) {
    var pct = (v * 100);
    return (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
  }

  /* ── Build all charts ── */
  function buildCharts() {
    charts.forEach(function (c) { c.destroy(); });
    charts = [];
    var C = upColors();

    buildEcon101(C);
    buildPriceDecomposition(C);
    buildAlphaIntervals(C);
    buildPipelineProvenance(C);
    buildSolverCurves(C);
    buildPiecewiseSupply(C);
    buildDualShiftFrontier(C);
  }

  /* ── 1. Stylized Econ-101 Supply & Demand Diagram ── */
  function buildEcon101(C) {
    var canvas = document.getElementById('econ101Chart');
    if (!canvas) return;

    // Curve functions (normalized 0-100 coordinate space)
    var SHIFT = 18; // horizontal demand shift for outsourcing
    function qSupply(p) { var t = Math.max(0, (p - 5) / 90); return 10 + 80 * Math.pow(t, 0.5); }
    function qDemand(p) { var t = Math.max(0, (p - 5) / 90); return 88 - 55 * Math.pow(t, 0.3); }
    function qDemandShifted(p) { return qDemand(p) + SHIFT; }

    // Generate curve data
    var nPts = 200;
    var supply = [], demand = [], demandShifted = [];
    for (var i = 0; i < nPts; i++) {
      var p = 5 + 92 * i / (nPts - 1);
      supply.push({ x: qSupply(p), y: p });
      demand.push({ x: qDemand(p), y: p });
      demandShifted.push({ x: qDemandShifted(p), y: p });
    }

    // Numerically find intersections by scanning for sign change
    function findIntersection(qA, qB) {
      var bestP = 50, bestGap = 1e9;
      for (var p = 6; p < 96; p += 0.1) {
        var gap = Math.abs(qA(p) - qB(p));
        if (gap < bestGap) { bestGap = gap; bestP = p; }
      }
      return { x: qA(bestP), y: bestP };
    }
    var E0 = findIntersection(qSupply, qDemand);
    var E1 = findIntersection(qSupply, qDemandShifted);

    charts.push(new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Supply (S)',
            data: supply,
            showLine: true,
            borderColor: C.teal,
            backgroundColor: 'transparent',
            borderWidth: 3,
            pointRadius: 0,
            tension: 0.4,
            order: 3
          },
          {
            label: 'Demand (D)',
            data: demand,
            showLine: true,
            borderColor: C.slateBlue,
            backgroundColor: 'transparent',
            borderWidth: 3,
            pointRadius: 0,
            tension: 0.4,
            order: 3
          },
          {
            label: 'Demand shifted (D\u2032)',
            data: demandShifted,
            showLine: true,
            borderColor: C.amber,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: [8, 4],
            pointRadius: 0,
            tension: 0.4,
            order: 3
          },
          {
            label: 'E\u2080  (P\u2080, Q\u2080)',
            data: [E0],
            pointRadius: 9,
            pointBackgroundColor: C.slateBlue,
            pointBorderColor: '#fff',
            pointBorderWidth: 2.5,
            showLine: false,
            order: 1
          },
          {
            label: 'E\u2032  (P(\u03B1), Q(\u03B1))',
            data: [E1],
            pointRadius: 9,
            pointBackgroundColor: C.amber,
            pointBorderColor: '#fff',
            pointBorderWidth: 2.5,
            pointStyle: 'triangle',
            showLine: false,
            order: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { left: 36, bottom: 4 } },
        plugins: {
          title: {
            display: true,
            text: 'Supply & Demand: Outsourcing Unpaid Childcare',
            font: { size: 15, weight: '700' },
            color: C.heading,
            padding: { bottom: 14 }
          },
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 18, font: { size: 12.5 } } },
          tooltip: { enabled: false }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: false },
            min: 0, max: 105,
            ticks: { display: false },
            grid: { display: false },
            border: { color: C.slate, width: 2 }
          },
          y: {
            title: { display: false },
            min: 0, max: 105,
            ticks: { display: false },
            grid: { display: false },
            border: { color: C.slate, width: 2 }
          }
        }
      },
      plugins: [{
        id: 'econ101Labels',
        afterDraw: function (chart) {
          var ctx = chart.ctx;
          var xScale = chart.scales.x;
          var yScale = chart.scales.y;
          ctx.save();

          // Dashed lines from E0 to axes
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = C.textMuted;
          ctx.lineWidth = 1;
          var e0x = xScale.getPixelForValue(E0.x);
          var e0y = yScale.getPixelForValue(E0.y);
          var e1x = xScale.getPixelForValue(E1.x);
          var e1y = yScale.getPixelForValue(E1.y);
          // E0 horizontal to y-axis
          ctx.beginPath(); ctx.moveTo(chart.chartArea.left, e0y); ctx.lineTo(e0x, e0y); ctx.stroke();
          // E0 vertical to x-axis
          ctx.beginPath(); ctx.moveTo(e0x, e0y); ctx.lineTo(e0x, chart.chartArea.bottom); ctx.stroke();
          // E1 horizontal to y-axis
          ctx.beginPath(); ctx.moveTo(chart.chartArea.left, e1y); ctx.lineTo(e1x, e1y); ctx.stroke();
          // E1 vertical to x-axis
          ctx.beginPath(); ctx.moveTo(e1x, e1y); ctx.lineTo(e1x, chart.chartArea.bottom); ctx.stroke();

          ctx.setLineDash([]);

          // Shaded region between E0 and E1 along supply curve
          ctx.fillStyle = C.tealLight || 'rgba(13,148,136,0.12)';
          ctx.beginPath();
          ctx.moveTo(e0x, chart.chartArea.bottom);
          ctx.lineTo(e0x, e0y);
          // trace supply curve from E0.y to E1.y
          var steps = 40;
          for (var si = 0; si <= steps; si++) {
            var pStep = E0.y + (E1.y - E0.y) * si / steps;
            var qStep = qSupply(pStep);
            ctx.lineTo(xScale.getPixelForValue(qStep), yScale.getPixelForValue(pStep));
          }
          ctx.lineTo(e1x, chart.chartArea.bottom);
          ctx.closePath();
          ctx.fill();

          // Labels
          ctx.font = '600 13px ' + Chart.defaults.font.family;
          // P0 label
          ctx.fillStyle = C.slateBlue;
          ctx.textAlign = 'right';
          ctx.fillText('P\u2080', chart.chartArea.left - 6, e0y + 4);
          // P(alpha) label
          ctx.fillStyle = C.amber;
          ctx.fillText('P(\u03B1)', chart.chartArea.left - 6, e1y + 4);
          // Q0 label
          ctx.fillStyle = C.slateBlue;
          ctx.textAlign = 'center';
          ctx.fillText('Q\u2080', e0x, chart.chartArea.bottom + 16);
          // Q(alpha) label
          ctx.fillStyle = C.amber;
          ctx.fillText('Q(\u03B1)', e1x, chart.chartArea.bottom + 16);

          // Axis endpoint labels
          ctx.font = '600 14px ' + Chart.defaults.font.family;
          ctx.fillStyle = C.slate;
          // "Quantity" at far right of x-axis
          ctx.textAlign = 'right';
          ctx.fillText('Quantity', chart.chartArea.right, chart.chartArea.bottom + 16);
          // "Price" at top of y-axis
          ctx.textAlign = 'left';
          ctx.fillText('Price', chart.chartArea.left + 4, chart.chartArea.top - 8);

          // Curve labels at curve endpoints using actual curve functions
          ctx.font = '700 15px ' + Chart.defaults.font.family;
          // S label near top-right of supply curve
          var sLabelP = 88;
          ctx.fillStyle = C.teal;
          ctx.textAlign = 'left';
          ctx.fillText('S', xScale.getPixelForValue(qSupply(sLabelP)) + 10, yScale.getPixelForValue(sLabelP) - 2);
          // D label near top-left of demand curve
          var dLabelP = 88;
          ctx.fillStyle = C.slateBlue;
          ctx.textAlign = 'right';
          ctx.fillText('D', xScale.getPixelForValue(qDemand(dLabelP)) - 8, yScale.getPixelForValue(dLabelP) - 2);
          // D' label near top of shifted demand curve — offset right to avoid overlapping the dashed line
          ctx.fillStyle = C.amber;
          ctx.textAlign = 'left';
          ctx.fillText('D\u2032', xScale.getPixelForValue(qDemandShifted(dLabelP)) + 10, yScale.getPixelForValue(dLabelP) - 2);

          // E0 label — offset left so it doesn't sit on the S curve
          ctx.font = '700 13px ' + Chart.defaults.font.family;
          ctx.fillStyle = C.heading;
          ctx.textAlign = 'right';
          ctx.fillText('E\u2080', e0x - 14, e0y - 6);
          // E' label — offset well above and right to clear the S curve
          ctx.textAlign = 'left';
          ctx.fillText('E\u2032', e1x + 16, e1y - 22);

          // Arrow showing demand shift — positioned between D and D' at a price below E0
          var arrowP = E0.y - 12; // below the equilibrium
          var arrowStartQ = qDemand(arrowP) + 2;
          var arrowEndQ = qDemandShifted(arrowP) - 2;
          var arrowYpx = yScale.getPixelForValue(arrowP);
          var arrowStartXpx = xScale.getPixelForValue(arrowStartQ);
          var arrowEndXpx = xScale.getPixelForValue(arrowEndQ);
          ctx.strokeStyle = C.amber;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(arrowStartXpx, arrowYpx);
          ctx.lineTo(arrowEndXpx, arrowYpx);
          ctx.stroke();
          // arrowhead
          ctx.beginPath();
          ctx.moveTo(arrowEndXpx, arrowYpx);
          ctx.lineTo(arrowEndXpx - 7, arrowYpx - 5);
          ctx.lineTo(arrowEndXpx - 7, arrowYpx + 5);
          ctx.closePath();
          ctx.fillStyle = C.amber;
          ctx.fill();
          // Arrow label
          ctx.font = '500 11px ' + Chart.defaults.font.family;
          ctx.fillStyle = C.amber;
          ctx.textAlign = 'center';
          ctx.fillText('\u03B1 shift', (arrowStartXpx + arrowEndXpx) / 2, arrowYpx - 10);

          ctx.restore();
        }
      }]
    }));
  }

  /* ── 2. Price Decomposition by Alpha ── */
  function buildPriceDecomposition(C) {
    var canvas = document.getElementById('priceDecompChart');
    if (!canvas) return;

    var labels = ['Baseline', '\u03B1=0.10', '\u03B1=0.25', '\u03B1=0.50', '\u03B1=1.00'];
    var gross = [8224, 8266, 8328, 8428, 8618];
    var directCare = [6551, 6594, 6631, 6691, 6831];
    var residual = gross.map(function (value, index) {
      return value - directCare[index];
    });
    var wages = ['$9.73/hr', '$9.81/hr', '$9.86/hr', '$9.95/hr', '$10.16/hr'];

    charts.push(new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Direct-care component',
            data: directCare,
            backgroundColor: C.teal,
            borderRadius: { topLeft: 0, topRight: 0, bottomLeft: 4, bottomRight: 4 },
            barPercentage: 0.55
          },
          {
            label: 'Displayed non-direct-care remainder',
            data: residual,
            backgroundColor: C.amber,
            borderRadius: { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 },
            barPercentage: 0.55
          }
        ]
      },
      options: {
        responsive: true,
        layout: { padding: { top: 25 } },
        plugins: {
          title: { display: true, text: 'Displayed price split by outsourcing share', font: { size: 15, weight: '700' }, color: C.heading, padding: { bottom: 8 } },
          subtitle: {
            display: true,
            text: 'Displayed gross and direct-care medians are rounded; the non-direct-care remainder is their arithmetic difference so the stacked bars close visually.',
            color: C.textSec,
            font: { size: 11.5, weight: '400' },
            padding: { bottom: 8 }
          },
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: function (items) {
                var idx = items[0].dataIndex;
                return labels[idx];
              },
              label: function (ctx) {
                var idx = ctx.dataIndex;
                if (ctx.datasetIndex === 0) {
                  return 'Displayed direct-care: ' + fmtDollar(directCare[idx]);
                }
                return 'Displayed non-direct-care remainder: ' + fmtDollar(residual[idx]);
              },
              afterBody: function (items) {
                var idx = items[0].dataIndex;
                return [
                  'Displayed gross: ' + fmtDollar(gross[idx]),
                  'Displayed residual = displayed gross minus displayed direct-care.',
                  'Implied wage: ' + wages[idx]
                ];
              }
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            grid: { display: false },
            ticks: { color: C.textSec, font: { size: 12 } }
          },
          y: {
            stacked: true,
            title: { display: true, text: 'Price per child per year ($)', color: C.textSec },
            ticks: { callback: function (v) { return fmtDollar(v); }, color: C.textSec },
            grid: { color: C.grid },
            beginAtZero: true
          }
        }
      },
      plugins: [{
        id: 'barLabels',
        afterDraw: function (chart) {
          var ctx = chart.ctx;
          var meta1 = chart.getDatasetMeta(1);
          ctx.save();
          ctx.textAlign = 'center';
          for (var i = 0; i < gross.length; i++) {
            var bar1 = meta1.data[i];
            // Gross above bar
            ctx.font = '700 12px ' + Chart.defaults.font.family;
            ctx.fillStyle = C.heading;
            ctx.fillText(fmtDollar(gross[i]), bar1.x, bar1.y - 8);
            // Wage below gross
            ctx.font = '500 10px ' + Chart.defaults.font.family;
            ctx.fillStyle = C.textMuted;
            ctx.fillText(wages[i], bar1.x, bar1.y - 22);
          }
          ctx.restore();
        }
      }]
    }));
  }

  /* ── 3. Scenario Intervals ── */
  function buildAlphaIntervals(C) {
    var canvas = document.getElementById('alphaIntervalsChart');
    if (!canvas) return;

    var labels = ['Marginal', '\u03B1=0.10', '\u03B1=0.25', '\u03B1=0.50', '\u03B1=1.00'];
    var medians = [8224, 8266, 8328, 8428, 8618];
    var lo = [8224, 8265, 8326, 8425, 8612];
    var hi = [8224, 8291, 8390, 8551, 8917];

    charts.push(new Chart(canvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Median marketization price',
            data: medians,
            borderColor: C.teal,
            backgroundColor: C.teal,
            pointRadius: 6,
            pointBackgroundColor: function (ctx) { return ctx.dataIndex === 0 ? C.amber : C.teal; },
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            borderWidth: 2.5,
            tension: 0.3,
            fill: false
          },
          {
            label: '10th\u201390th percentile',
            data: hi,
            borderColor: 'transparent',
            backgroundColor: C.tealLight || (C.teal + '22'),
            pointRadius: 0,
            borderWidth: 0,
            fill: '+1',
            tension: 0.3
          },
          {
            label: '_lo',
            data: lo,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            borderWidth: 0,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: { display: true, text: 'Canonical Scenario Intervals', font: { size: 15, weight: '700' }, color: C.heading, padding: { bottom: 8 } },
          legend: {
            position: 'bottom',
            labels: {
              filter: function (item) { return item.text !== '_lo'; },
              padding: 16
            }
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.datasetIndex === 0) {
                  var idx = ctx.dataIndex;
                  return 'Median: ' + fmtDollar(medians[idx]) + ' [' + fmtDollar(lo[idx]) + '\u2013' + fmtDollar(hi[idx]) + ']';
                }
                return null;
              }
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'Marketization price ($)', color: C.textSec },
            ticks: { callback: function (v) { return fmtDollar(v); }, color: C.textSec },
            grid: { color: C.grid }
          },
          x: {
            grid: { display: false },
            ticks: { color: C.textSec }
          }
        }
      },
      plugins: [{
        id: 'baselineLine',
        beforeDraw: function (chart) {
          var yScale = chart.scales.y;
          var ctx = chart.ctx;
          var yPx = yScale.getPixelForValue(8224);
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = C.textMuted;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chart.chartArea.left, yPx);
          ctx.lineTo(chart.chartArea.right, yPx);
          ctx.stroke();
          ctx.restore();
          ctx.save();
          ctx.font = '500 11px ' + Chart.defaults.font.family;
          ctx.fillStyle = C.textMuted;
          ctx.textAlign = 'right';
          ctx.fillText('Baseline $8,224', chart.chartArea.right - 4, yPx - 6);
          ctx.restore();
        }
      }]
    }));
  }

  /* ── 4. Pipeline Provenance ── */
  function buildPipelineProvenance(C) {
    var canvas = document.getElementById('provenanceChart');
    if (!canvas) return;

    var labels = ['State births', 'State controls', 'County ACS', 'County wages', 'County jobs', 'County LAUS'];
    var observed = [66, 55, 96, 62, 62, 100];

    charts.push(new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Observed coverage',
            data: observed,
            backgroundColor: C.teal,
            borderRadius: 4,
            barPercentage: 0.6
          },
          {
            label: 'Gap (synthetic/fallback)',
            data: observed.map(function (v) { return 100 - v; }),
            backgroundColor: C.muted,
            borderRadius: 4,
            barPercentage: 0.6
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: {
          title: { display: true, text: 'Pipeline Provenance: Observed-Source Coverage', font: { size: 15, weight: '700' }, color: C.heading, padding: { bottom: 8 } },
          legend: { position: 'bottom', labels: { padding: 16 } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': ' + ctx.parsed.x + '%';
              }
            }
          }
        },
        scales: {
          x: {
            stacked: true,
            max: 100,
            title: { display: true, text: 'Coverage (%)', color: C.textSec },
            ticks: { callback: function (v) { return v + '%'; }, color: C.textSec },
            grid: { color: C.grid }
          },
          y: {
            stacked: true,
            grid: { display: false },
            ticks: { color: C.textSec }
          }
        }
      }
    }));
  }

  /* ── 5. Solver-Implied Curves ── */
  function buildSolverCurves(C) {
    var canvas = document.getElementById('solverCurvesChart');
    if (!canvas) return;

    var P0 = 8224, Q0 = 3.18e6;
    var ed = -0.143, es = 4.035;
    var pMin = 7400, pMax = 9000;

    var supply = generateCurve(P0, Q0, es, pMin, pMax, 200);
    var demand = generateCurve(P0, Q0, ed, pMin, pMax, 200);

    var pA50 = 8428, qA50 = Q0 * Math.pow(pA50 / P0, es);
    var pA100 = 8618, qA100 = Q0 * Math.pow(pA100 / P0, es);

    charts.push(new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Supply (\u03B5 = 4.035)',
            data: supply,
            showLine: true,
            borderColor: C.teal,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.4
          },
          {
            label: 'Demand (\u03B5 = \u22120.143)',
            data: demand,
            showLine: true,
            borderColor: C.slateBlue,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.4
          },
          {
            label: 'Baseline',
            data: [{ x: Q0, y: P0 }],
            pointRadius: 8,
            pointBackgroundColor: C.heading,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            showLine: false
          },
          {
            label: '\u03B1 = 0.50 (+$204)',
            data: [{ x: qA50, y: pA50 }],
            pointRadius: 7,
            pointBackgroundColor: C.amber,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointStyle: 'triangle',
            showLine: false
          },
          {
            label: '\u03B1 = 1.00 (+$394)',
            data: [{ x: qA100, y: pA100 }],
            pointRadius: 7,
            pointBackgroundColor: C.red,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointStyle: 'rectRot',
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          title: { display: true, text: 'Solver-Implied Supply & Demand Curves', font: { size: 15, weight: '700' }, color: C.heading, padding: { bottom: 12 } },
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': ' + fmtDollar(Math.round(ctx.parsed.y)) + ' / ' + fmtM(ctx.parsed.x);
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Quantity of paid childcare', color: C.textSec },
            min: 1.5e6, max: 4.0e6,
            ticks: { callback: function (v) { return fmtM(v); }, color: C.textSec },
            grid: { color: C.grid }
          },
          y: {
            title: { display: true, text: 'Price per child-year ($)', color: C.textSec },
            min: 7400, max: 9000,
            ticks: { callback: function (v) { return fmtDollar(v); }, color: C.textSec },
            grid: { color: C.grid }
          }
        }
      }
    }));
  }

  /* ── 6. Piecewise Supply Demo ── */
  function buildPiecewiseSupply(C) {
    var canvas = document.getElementById('piecewiseChart');
    if (!canvas) return;

    var P0 = 8224, Q0 = 3.18e6;
    var etaBelow = 5.776, etaAbove = 2.595, etaConst = 3.906;
    var pMin = 6500, pMax = 11000;

    // Constant-elasticity supply
    var constSupply = generateCurve(P0, Q0, etaConst, pMin, pMax, 200);

    // Piecewise supply: different elasticity below and above P0
    var pwSupply = [];
    for (var i = 0; i < 200; i++) {
      var p = pMin + (pMax - pMin) * i / 199;
      var eta = p <= P0 ? etaBelow : etaAbove;
      var q = Q0 * Math.pow(p / P0, eta);
      pwSupply.push({ x: q, y: p });
    }

    charts.push(new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Constant supply (\u03B5 = 3.91)',
            data: constSupply,
            showLine: true,
            borderColor: C.teal,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.4
          },
          {
            label: 'Piecewise supply (\u03B7\u2093 = 5.78 / \u03B7\u2090 = 2.60)',
            data: pwSupply,
            showLine: true,
            borderColor: C.amber,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: [6, 3],
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: 'Baseline (P\u2080)',
            data: [{ x: Q0, y: P0 }],
            pointRadius: 8,
            pointBackgroundColor: C.heading,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Piecewise vs. Constant Supply', font: { size: 15, weight: '700' }, color: C.heading, padding: { bottom: 12 } },
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 12 } } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ': ' + fmtDollar(Math.round(ctx.parsed.y)) + ' / ' + fmtM(ctx.parsed.x);
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Quantity of paid childcare', color: C.textSec },
            ticks: { callback: function (v) { return fmtM(v); }, color: C.textSec },
            grid: { color: C.grid }
          },
          y: {
            title: { display: true, text: 'Price per child-year ($)', color: C.textSec },
            min: 6500, max: 11000,
            ticks: { callback: function (v) { return fmtDollar(v); }, color: C.textSec },
            grid: { color: C.grid }
          }
        }
      },
      plugins: [{
        id: 'baselineKink',
        afterDraw: function (chart) {
          var yScale = chart.scales.y;
          var ctx = chart.ctx;
          var yPx = yScale.getPixelForValue(P0);
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = C.textMuted;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chart.chartArea.left, yPx);
          ctx.lineTo(chart.chartArea.right, yPx);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = '500 11px ' + Chart.defaults.font.family;
          ctx.fillStyle = C.textMuted;
          ctx.textAlign = 'left';
          ctx.fillText('Kink at P\u2080 = $8,224', chart.chartArea.left + 6, yPx - 6);
          ctx.restore();
        }
      }]
    }));
  }

  /* ── 7. Dual-shift Marketization Frontier ── */
  function buildDualShiftFrontier(C) {
    var canvas = document.getElementById('dualShiftChart');
    if (!canvas) return;
    var headlineAlpha = 0.50;

    var rawPoints = [
      { kappaC: 0.00, kappaQ: 0.00, pct: 0.054941500406918704, price: 8726.578698035715 },
      { kappaC: 0.05, kappaQ: 0.00, pct: 0.08099622257993914, price: 8942.249864276215 },
      { kappaC: 0.10, kappaQ: 0.00, pct: 0.1076948430111881, price: 9163.254801422605 },
      { kappaC: 0.15, kappaQ: 0.00, pct: 0.13505328531029068, price: 9389.72551482762 },
      { kappaC: 0.20, kappaQ: 0.00, pct: 0.1630878671629812, price: 9621.79727936487 },
      { kappaC: 0.00, kappaQ: 0.25, pct: 0.02385369198823952, price: 8469.250483636068 },
      { kappaC: 0.05, kappaQ: 0.25, pct: 0.04914014561587713, price: 8678.557780774307 },
      { kappaC: 0.10, kappaQ: 0.25, pct: 0.0750514985822997, price: 8893.041352855675 },
      { kappaC: 0.15, kappaQ: 0.25, pct: 0.10160320432324998, price: 9112.82930444388 },
      { kappaC: 0.20, kappaQ: 0.25, pct: 0.1288110987060361, price: 9338.052912931602 },
      { kappaC: 0.00, kappaQ: 0.50, pct: -0.006317454742134933, price: 8219.515181110375 },
      { kappaC: 0.05, kappaQ: 0.50, pct: 0.018223398513271954, price: 8422.646518139401 },
      { kappaC: 0.10, kappaQ: 0.50, pct: 0.043370713154811655, price: 8630.801288091963 },
      { kappaC: 0.15, kappaQ: 0.50, pct: 0.06913948633720368, price: 8844.103810101053 },
      { kappaC: 0.20, kappaQ: 0.50, pct: 0.09554508634653812, price: 9062.681482298554 },
      { kappaC: 0.00, kappaQ: 0.75, pct: -0.03559898586754299, price: 7977.14859543913 },
      { kappaC: 0.05, kappaQ: 0.75, pct: -0.011781734103867694, price: 8174.286328834152 },
      { kappaC: 0.10, kappaQ: 0.75, pct: 0.012624085484792219, price: 8376.299169448022 },
      { kappaC: 0.15, kappaQ: 0.75, pct: 0.03763302725416548, price: 8583.307762927026 },
      { kappaC: 0.20, kappaQ: 0.75, pct: 0.06326000572531515, price: 8795.435742863374 },
      { kappaC: 0.00, kappaQ: 1.00, pct: -0.06401714893399774, price: 7741.93315619452 },
      { kappaC: 0.05, kappaQ: 1.00, pct: -0.04090214929856241, price: 7933.2542541105195 },
      { kappaC: 0.10, kappaQ: 1.00, pct: -0.017215947092932886, price: 8129.306516300609 },
      { kappaC: 0.15, kappaQ: 1.00, pct: 0.0070555823229907924, price: 8330.207023555253 },
      { kappaC: 0.20, kappaQ: 1.00, pct: 0.0319269131128765, price: 8536.075756252707 },
      { kappaC: 0.00, kappaQ: 1.25, pct: -0.09159741654413969, price: 7513.657721644142 },
      { kappaC: 0.05, kappaQ: 1.25, pct: -0.06916394999957386, price: 7699.333923253931 },
      { kappaC: 0.10, kappaQ: 1.25, pct: -0.04617613344276039, price: 7889.601599350652 },
      { kappaC: 0.15, kappaQ: 1.25, pct: -0.022620259252736568, price: 8084.574371284845 },
      { kappaC: 0.20, kappaQ: 1.25, pct: 0.0015177193879351647, price: 8284.368674251695 },
      { kappaC: 0.00, kappaQ: 1.50, pct: -0.1183645092535323, price: 7292.117388652263 },
      { kappaC: 0.05, kappaQ: 1.50, pct: -0.09659246846515691, price: 7472.315358768932 },
      { kappaC: 0.10, kappaQ: 1.50, pct: -0.07428243267359438, price: 7656.969240976212 },
      { kappaC: 0.15, kappaQ: 1.50, pct: -0.051421098955947465, price: 7846.189299480853 },
      { kappaC: 0.20, kappaQ: 1.50, pct: -0.027994835214578337, price: 8040.088529128645 }
    ];

    var rawFrontier = [
      { kappaC: 0.00, kappaQStar: 0.500929115647994 },
      { kappaC: 0.05, kappaQStar: 0.7048404721048158 },
      { kappaC: 0.10, kappaQStar: 0.9087518285616376 },
      { kappaC: 0.15, kappaQStar: 1.1126631850184594 },
      { kappaC: 0.20, kappaQStar: 1.3165745414752812 }
    ];

    function multiplierFromKappa(kappa) {
      return Math.exp(headlineAlpha * kappa) - 1;
    }

    function fmtPercentAxis(value) {
      return Math.round(value * 100) + '%';
    }

    var points = rawPoints.map(function (point) {
      return {
        x: multiplierFromKappa(point.kappaC),
        y: multiplierFromKappa(point.kappaQ),
        pct: point.pct,
        price: point.price,
        kappaC: point.kappaC,
        kappaQ: point.kappaQ
      };
    });

    var frontier = rawFrontier.map(function (point) {
      return {
        x: multiplierFromKappa(point.kappaC),
        y: multiplierFromKappa(point.kappaQStar),
        kappaC: point.kappaC,
        kappaQStar: point.kappaQStar
      };
    });

    function pointColor(pct) {
      if (pct < 0) {
        if (pct <= -0.05) return C.slateBlue;
        if (pct <= -0.02) return '#60a5fa';
        return '#93c5fd';
      }
      if (pct < 0.03) return '#86efac';
      if (pct < 0.08) return C.amber;
      return C.red;
    }

    charts.push(new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Simulated combinations',
            data: points,
            pointRadius: 12,
            pointHoverRadius: 14,
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointBackgroundColor: points.map(function (point) { return pointColor(point.pct); }),
            showLine: false,
            order: 2
          },
          {
            label: 'Median zero-price frontier',
            data: frontier,
            showLine: true,
            borderColor: C.heading,
            backgroundColor: C.heading,
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 4,
            pointHoverRadius: 5,
            order: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Medium-run marketization at alpha = 0.50: when does price rise, stay flat, or fall?',
            font: { size: 15, weight: '700' },
            color: C.heading,
            padding: { bottom: 12 }
          },
          subtitle: {
            display: true,
            text: 'Each dot shows the median price change when more unpaid care is shifted into paid care and the market also changes on the provider side.',
            color: C.textSec,
            font: { size: 11.5, weight: '400' },
            padding: { bottom: 8 }
          },
          legend: {
            position: 'bottom',
            labels: { usePointStyle: true, padding: 16, font: { size: 12 } }
          },
          tooltip: {
            callbacks: {
              title: function (items) {
                var item = items[0];
                if (item.datasetIndex === 1) {
                  return 'Price stays about flat';
                }
                return 'Cost pressure ' + fmtPercentAxis(item.raw.x) + ', paid-care capacity ' + fmtPercentAxis(item.raw.y);
              },
              label: function (ctx) {
                if (ctx.datasetIndex === 1) {
                  return [
                    'Extra paid-care capacity needed: +' + fmtPercentAxis(ctx.raw.y),
                    'Provider cost increase: ' + fmtPercentAxis(ctx.raw.x)
                  ];
                }
                return [
                  'Median price change: ' + fmtPct(ctx.raw.pct),
                  'Median price: ' + fmtDollar(Math.round(ctx.raw.price))
                ];
              }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            min: -0.01,
            max: 0.115,
            title: { display: true, text: 'Provider cost increase at alpha = 0.50', color: C.textSec },
            ticks: {
              stepSize: 0.05,
              color: C.textSec,
              callback: function (v) { return fmtPercentAxis(Number(v)); }
            },
            grid: { color: C.grid }
          },
          y: {
            min: -0.02,
            max: 1.15,
            title: { display: true, text: 'Paid-care capacity expansion at alpha = 0.50', color: C.textSec },
            ticks: {
              stepSize: 0.25,
              color: C.textSec,
              callback: function (v) { return fmtPercentAxis(Number(v)); }
            },
            grid: { color: C.grid }
          }
        }
      },
      plugins: [{
        id: 'dualShiftLabels',
        afterDatasetsDraw: function (chart) {
          var ctx = chart.ctx;
          var meta = chart.getDatasetMeta(0);
          ctx.save();
          ctx.font = '600 10px ' + Chart.defaults.font.family;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          meta.data.forEach(function (element, index) {
            var point = points[index];
            ctx.fillStyle = Math.abs(point.pct) >= 0.05 ? '#fff' : C.heading;
            ctx.fillText(fmtPct(point.pct), element.x, element.y);
          });
          ctx.restore();
        }
      }]
    }));
  }

  /* ── Master rebuild ── */
  window.upRebuildCharts = function () {
    upChartDefaults();
    buildCharts();
  };

  /* ── Init on load ── */
  document.addEventListener('DOMContentLoaded', function () {
    upChartDefaults();
    buildCharts();
    upInitScrollSpy();
    upInitCollapsibles();
  });
})();
