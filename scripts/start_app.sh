#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8511
HOST=127.0.0.1
DETACH=0
CMD_PID_FILE="$PROJECT_DIR/.start_app.pid"
LOG_FILE="$PROJECT_DIR/.streamlit.log"
SCREEN_NAME_PREFIX="fraud-risk-app"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/start_app.sh [--bg] [PORT] [HOST]
  ./scripts/start_app.sh --status
  ./scripts/start_app.sh --stop

Examples:
  ./scripts/start_app.sh 8511 127.0.0.1
  ./scripts/start_app.sh --bg
  ./scripts/start_app.sh --stop
EOF
}

resolve_streamlit() {
  if [ -x "$PROJECT_DIR/.venv/bin/streamlit" ]; then
    STREAMLIT_BIN="$PROJECT_DIR/.venv/bin/streamlit"
  elif [ -x "/Users/igna/entorno/bin/streamlit" ]; then
    STREAMLIT_BIN="/Users/igna/entorno/bin/streamlit"
  elif command -v streamlit >/dev/null 2>&1; then
    STREAMLIT_BIN="streamlit"
  else
    echo "No se encontró streamlit en .venv, /Users/igna/entorno ni en PATH." >&2
    exit 1
  fi
}

cleanup_pid() {
  if [ -f "$CMD_PID_FILE" ] && [ "$(cat "$CMD_PID_FILE" 2>/dev/null || echo)" = "$1" ]; then
    rm -f "$CMD_PID_FILE"
  fi
}

kill_pid() {
  local target_pid="$1"
  if [ -n "$target_pid" ] && ps -p "$target_pid" >/dev/null 2>&1; then
    kill "$target_pid" >/dev/null 2>&1 || true
  fi
}

screen_name() {
  printf "%s-%s" "$SCREEN_NAME_PREFIX" "$PORT"
}

status() {
  if [ -f "$CMD_PID_FILE" ]; then
    pid="$(cat "$CMD_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      echo "APP ejecutándose. PID: $pid"
      return 0
    fi
  fi

  if pgrep -f "streamlit run app.py" >/dev/null 2>&1; then
    echo "APP ejecutándose (Streamlit encontrado sin PID guardado)."
    return 0
  fi

  if command -v screen >/dev/null 2>&1 && screen -ls | grep -q "[.]$(screen_name)[[:space:]]"; then
    echo "APP ejecutándose en screen: $(screen_name)"
    return 0
  fi

  echo "APP no está en ejecución."
  return 1
}

stop() {
  if [ ! -f "$CMD_PID_FILE" ]; then
    echo "No hay PID guardado."
  fi

  pid="$(cat "$CMD_PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
    echo "Deteniendo PID ${pid}..."
    kill_pid "$pid"
  fi

  for streamlit_pid in $(pgrep -f "streamlit run app.py" || true); do
    echo "Deteniendo Streamlit PID ${streamlit_pid}..."
    kill "$streamlit_pid" >/dev/null 2>&1 || true
  done

  for streamlit_pid in $(lsof -nP -sTCP:LISTEN -iTCP:"${PORT}" -t 2>/dev/null || true); do
    if ps -p "$streamlit_pid" -o command= | grep -q "streamlit"; then
      echo "Liberando puerto ${PORT} desde PID ${streamlit_pid}..."
      kill "$streamlit_pid" >/dev/null 2>&1 || true
    fi
  done

  if command -v screen >/dev/null 2>&1; then
    screen -S "$(screen_name)" -X quit >/dev/null 2>&1 || true
  fi

  rm -f "$CMD_PID_FILE"
  echo "Detenido."
  return 0
}

while (($# > 0)); do
  case "${1:-}" in
    --bg|-b)
      DETACH=1
      shift
      ;;
    --status)
      status
      exit $?
      ;;
    --stop)
      stop
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        PORT="$1"
      else
        HOST="$1"
      fi
      shift
      ;;
  esac
done

cd "$PROJECT_DIR"
resolve_streamlit

if lsof -iTCP -sTCP:LISTEN -nP -iTCP:"${PORT}" >/dev/null; then
  for pid in $(lsof -nP -sTCP:LISTEN -iTCP:"${PORT}" -t); do
    if ps -p "$pid" -o command= | grep -q "streamlit"; then
      echo "Puerto ${PORT} ocupado por Streamlit en PID ${pid}. Cerrando..."
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  sleep 1
fi

STREAMLIT_CMD=(
  "$STREAMLIT_BIN" run app.py --server.address "${HOST}" --server.port "${PORT}" --server.headless true
)

echo "Iniciando aplicación:"
echo "  ${STREAMLIT_CMD[*]}"
echo "URL principal: http://localhost:${PORT}"
echo "URL alternativa: http://${HOST}:${PORT}"
echo "Log: ${LOG_FILE}"

if (( DETACH == 1 )); then
  if command -v screen >/dev/null 2>&1; then
    screen -S "$(screen_name)" -X quit >/dev/null 2>&1 || true
    screen -dmS "$(screen_name)" bash -lc "cd \"$PROJECT_DIR\" && exec \"$STREAMLIT_BIN\" run app.py --server.address \"$HOST\" --server.port \"$PORT\" --server.headless true >> \"$LOG_FILE\" 2>&1"
  else
    nohup "${STREAMLIT_CMD[@]}" < /dev/null >> "$LOG_FILE" 2>&1 &
    echo "$!" > "$CMD_PID_FILE"
  fi

  sleep 3
  if ! lsof -iTCP -sTCP:LISTEN -nP -iTCP:"${PORT}" >/dev/null 2>&1; then
    echo "No se pudo confirmar que Streamlit quede activo en el puerto ${PORT}."
    echo "Revisá el log:"
    tail -n 30 "$LOG_FILE"
    if command -v screen >/dev/null 2>&1; then
      screen -S "$(screen_name)" -X quit >/dev/null 2>&1 || true
    fi
    rm -f "$CMD_PID_FILE"
    exit 1
  fi

  app_pid="$(lsof -nP -sTCP:LISTEN -iTCP:"${PORT}" -t | head -n 1)"
  echo "$app_pid" > "$CMD_PID_FILE"
  echo "App iniciada en segundo plano con PID ${app_pid}"
  exit 0
fi

trap 'cleanup_pid "$$"' EXIT INT TERM
echo $$ > "$CMD_PID_FILE"
"${STREAMLIT_CMD[@]}" | tee "$LOG_FILE"
