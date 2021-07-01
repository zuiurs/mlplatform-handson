# Manifests v1.3

ベースのマニフェストの生成。

```bash
git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.3.0
kustomize build example > manifests.yaml
```

マニフェストは CRD とそうでないものに分離済み。まとめて適用すると CRD 周りでコケるので分けて適用する。

```bash
kubectl apply -f crd.yaml
kubectl apply -f resource.yaml
```

## Patch

TLS の設定をしないのでリソース側にちょっとだけ修正をしている。

```bash
kubectl edit destinationrule -n kubeflow ml-pipeline
# Modify the tls.mode (the last line) from ISTIO_MUTUAL to DISABLE

kubectl edit destinationrule -n kubeflow ml-pipeline-ui
# Modify the tls.mode (the last line) from ISTIO_MUTUAL to DISABLE
```

## Authorization Policy

まれに `istio-ingressgateway` と `kubeflow` 間のポリシー制御がおかしくなり Pipeline 系の UI が見えなくなることがあるので、ハンズオンの環境では全許可をしている。

```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-all
  namespace: kubeflow
spec:
 rules:
 - {}
```

追記: knative-serving の Activator でも同様のことが起こり、急ぎの対処として同じく全許可ポリシーを当てている。
