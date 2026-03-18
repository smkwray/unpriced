/* unpriced — theme, Chart.js colors */

(function () {
  'use strict';

  function getSavedTheme() {
    try { return localStorage.getItem('unpriced-theme'); } catch (e) { return null; }
  }
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('unpriced-theme', theme); } catch (e) {}
    var btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = theme === 'dark' ? '\u2600' : '\u263E';
  }

  applyTheme(getSavedTheme() || 'light');

  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
    if (!getSavedTheme()) applyTheme(e.matches ? 'dark' : 'light');
  });

  window.upToggleTheme = function () {
    var cur = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(cur === 'dark' ? 'light' : 'dark');
    if (window.upRebuildCharts) window.upRebuildCharts();
  };

  window.upIsDark = function () {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  };

  window.upColors = function () {
    var dk = window.upIsDark();
    return {
      teal:      dk ? '#2dd4bf' : '#0d9488',
      tealLight: dk ? 'rgba(45,212,191,0.18)' : 'rgba(13,148,136,0.12)',
      amber:     dk ? '#fbbf24' : '#d97706',
      amberLight:dk ? 'rgba(251,191,36,0.18)' : 'rgba(217,119,6,0.12)',
      slate:     dk ? '#94a3b8' : '#475569',
      slateBlue: dk ? '#7dd3fc' : '#3b82f6',
      red:       dk ? '#f87171' : '#dc2626',
      redLight:  dk ? 'rgba(248,113,113,0.15)' : 'rgba(220,38,38,0.10)',
      green:     dk ? '#4ade80' : '#16a34a',
      greenLight:dk ? 'rgba(74,222,128,0.15)' : 'rgba(22,163,74,0.10)',
      text:      dk ? '#e2e8f0' : '#1e293b',
      textSec:   dk ? '#94a3b8' : '#64748b',
      textMuted: dk ? '#64748b' : '#94a3b8',
      heading:   dk ? '#f1f5f9' : '#0f172a',
      bg:        dk ? '#0f172a' : '#ffffff',
      bgAlt:     dk ? '#1e293b' : '#f8fafc',
      bgCard:    dk ? '#1e293b' : '#ffffff',
      border:    dk ? '#334155' : '#e2e8f0',
      borderLight: dk ? '#1e293b' : '#f1f5f9',
      grid:      dk ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
      muted:     dk ? '#334155' : '#cbd5e1',
      navy:      dk ? '#0f172a' : '#0f172a',
      palette: dk
        ? ['#2dd4bf','#7dd3fc','#fbbf24','#f87171','#a78bfa','#94a3b8','#4ade80','#fb923c']
        : ['#0d9488','#3b82f6','#d97706','#dc2626','#7c3aed','#475569','#16a34a','#ea580c']
    };
  };

  window.upChartDefaults = function () {
    if (typeof Chart === 'undefined') return;
    try {
      var c = window.upColors();
      Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      Chart.defaults.font.size = 13;
      Chart.defaults.color = c.textSec;
      Chart.defaults.borderColor = c.grid;
      if (Chart.defaults.plugins.legend && Chart.defaults.plugins.legend.labels)
        Chart.defaults.plugins.legend.labels.color = c.textSec;
      if (Chart.defaults.plugins.tooltip) {
        Chart.defaults.plugins.tooltip.backgroundColor = c.navy;
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        Chart.defaults.plugins.tooltip.bodyColor = '#e0e0e0';
        Chart.defaults.plugins.tooltip.cornerRadius = 6;
        Chart.defaults.plugins.tooltip.padding = 10;
      }
    } catch (e) {}
  };

  /* ── Smooth scroll for nav ── */
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-links a').forEach(function (a) {
      a.addEventListener('click', function () {
        var nl = document.querySelector('.nav-links');
        if (nl) nl.classList.remove('open');
      });
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        var nl = document.querySelector('.nav-links');
        if (nl) nl.classList.remove('open');
      }
    });
  });

  /* ── Scroll-spy ── */
  window.upInitScrollSpy = function () {
    var secs = document.querySelectorAll('section[id]');
    var links = document.querySelectorAll('.nav-links a[href^="#"]');
    if (!secs.length || !links.length) return;
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          var t = document.querySelector('.nav-links a[href="#' + e.target.id + '"]');
          if (t) t.classList.add('active');
        }
      });
    }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });
    secs.forEach(function (s) { obs.observe(s); });
  };

  /* ── Collapsible sections ── */
  window.upInitCollapsibles = function () {
    document.querySelectorAll('.collapsible-header').forEach(function (h) {
      h.addEventListener('click', function () {
        this.parentElement.classList.toggle('open');
      });
    });
  };
})();
