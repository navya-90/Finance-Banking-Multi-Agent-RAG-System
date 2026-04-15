import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor
from config import PHOENIX_PORT


def setup_tracing():
    # Launch Phoenix UI (opens at http://localhost:{PHOENIX_PORT})
    px.launch_app(port=PHOENIX_PORT)

    # Wire OpenTelemetry → Phoenix exporter
    from phoenix.otel import register
    tracer_provider = register(
        project_name="finance-banking-agents",
        auto_instrument=False,
    )

    # Auto-instrument all LangChain / LangGraph calls
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"Phoenix tracing UI: http://localhost:{PHOENIX_PORT}")
    return tracer_provider


def get_tracer(name: str = "finance-agents"):
    return trace.get_tracer(name)
