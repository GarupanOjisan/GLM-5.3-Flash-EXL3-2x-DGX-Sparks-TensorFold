FROM tensorfold-glm53:k3p
COPY carveout.py carveout_export.py k3c_patch.py /opt/carveout/
RUN python /opt/carveout/k3c_patch.py
