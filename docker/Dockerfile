FROM python:3.10-slim

# Combine all installation and cleanup steps into a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget unzip && \
    wget https://github.com/amarquand/PCNtoolkit/archive/dev.zip && \
    unzip dev.zip && \
    cd PCNtoolkit-dev && python -m pip install . && cd .. && \
    # Cleanup
    rm -rf PCNtoolkit-dev dev.zip && \
    pip cache purge && \
    apt-get remove -y wget unzip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]