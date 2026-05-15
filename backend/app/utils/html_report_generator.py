from datetime import datetime, timezone
from html import escape
from typing import Any, Optional

from jinja2 import Template

DEFAULT_REPORT_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет КТ аорты</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      color: #111827;
      margin: 32px;
    }
    h1, h2 {
      margin: 0 0 12px;
    }
    h1 {
      font-size: 22px;
      border-bottom: 2px solid #111827;
      padding-bottom: 8px;
    }
    h2 {
      font-size: 17px;
      margin-top: 24px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }
    th, td {
      border: 1px solid #d1d5db;
      padding: 8px;
      vertical-align: top;
    }
    th {
      background: #f3f4f6;
      text-align: left;
    }
    .meta {
      margin-top: 16px;
    }
    .muted {
      color: #6b7280;
      font-size: 12px;
    }
    .section {
      margin-top: 18px;
    }
    pre {
      white-space: pre-wrap;
      font-family: Arial, sans-serif;
    }
  </style>
</head>
<body>
  <h1>Заключение врача-кардиохирурга по данным КТ</h1>

  <div class="meta">
    <table>
      <tr>
        <th>Пациент</th>
        <td>{{ meta.name }}</td>
      </tr>
      <tr>
        <th>Пол</th>
        <td>{{ meta.sex }}</td>
      </tr>
      <tr>
        <th>Дата рождения</th>
        <td>{{ meta.birth_date }}</td>
      </tr>
      <tr>
        <th>Дата КТ</th>
        <td>{{ meta.ct_date }}</td>
      </tr>
      <tr>
        <th>Анамнез</th>
        <td>{{ meta.anamnesis }}</td>
      </tr>
    </table>
  </div>

  <h2>Ключевые измерения</h2>
  <table>
    <thead>
      <tr>
        <th>Параметр</th>
        <th>Значение</th>
      </tr>
    </thead>
    <tbody>
      {% for key, value in measurements.items() %}
      <tr>
        <td>{{ key }}</td>
        <td>{{ value }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Сформированное заключение</h2>
  <div class="section">
    <pre>{{ llm_response }}</pre>
  </div>

  <p class="muted">
    Отчет сформирован автоматически. Требуется проверка врачом.
    Дата формирования: {{ generated_at }}
  </p>
</body>
</html>
"""

def _as_dict(value: Any) -> dict:
    if isinstance(value,dict):
        return value
    return {}

def _normalize_llm_response(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value,str):
        return value

    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}: {item}")
        return "\n\n".join(parts)
    
    return str(value)

def generate_html_report(report, template_content: Optional[str] = None) -> str:
    
    template = Template(template_content or DEFAULT_REPORT_TEMPLATE)

    meta = _as_dict(report.meta)
    measurements = _as_dict(report.measurements)
    llm_response = _normalize_llm_response(report.llm_response)
    
    return template.render(
        meta=meta,
        measurements=measurements,
        llm_response = llm_response,
        report=report,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )