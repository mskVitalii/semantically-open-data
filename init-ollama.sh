#!/bin/sh
set -e

# Запускаем Ollama сервер
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    max_attempts=30
    attempt=0

    sleep 10

    while [ $attempt -lt $max_attempts ]; do
        if ollama list >/dev/null 2>&1; then
            echo "Ollama is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts - Ollama not ready yet..."
        sleep 3
    done

    echo "Ollama failed to start after $max_attempts attempts"
    return 1
}

pull_model() {
    model=$1
    echo "Checking if model '$model' exists..."

    if ollama list 2>/dev/null | grep -q "^$model"; then
        echo "Model '$model' already exists"
    else
        echo "Pulling model '$model'..."
        if ollama pull "$model"; then
            echo "Model '$model' pulled successfully"
        else
            echo "Failed to pull model '$model'"
            return 1
        fi
    fi
}

if wait_for_ollama; then
    if [ -n "$OLLAMA_MODELS" ]; then
        IFS=','
        for model in $OLLAMA_MODELS; do
            model=$(echo "$model" | tr -d ' ')
            if [ -n "$model" ]; then
                pull_model "$model" || true
            fi
        done
    else
        pull_model "llama2"
    fi
else
    echo "Ollama server failed to start"
    exit 1
fi

echo "Ollama is running with PID $OLLAMA_PID"
wait $OLLAMA_PID
