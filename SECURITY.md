# Security Policy

## Supported Versions

이 프로젝트는 활발히 개발 중이며, 현재는 최신 릴리스만 보안 패치를 지원합니다.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest | :x:                |

---

## 배포 보안

- **패키지 서명**: PyPI에 배포되는 모든 릴리스는 [Sigstore](https://www.sigstore.dev/)로 서명됩니다. `pip install` 시 서명을 검증할 수 있습니다.
- **의존성 취약점 스캔**: 모든 push/PR에서 GitHub Actions가 `pip-audit`을 실행해 알려진 취약점이 있는 의존성을 자동으로 감지합니다. 저장소의 Dependabot 취약점 알림(vulnerability alerts)도 켜져 있어 새로운 CVE가 등록되면 알림을 받습니다(자동 업데이트 PR 생성은 아직 설정하지 않음).
- **SBOM(Software Bill of Materials)**: 릴리스마다 CycloneDX 형식의 SBOM이 자동 생성되어, 기업 환경에서 의존성 목록을 감사할 때 활용할 수 있습니다.

---

## 취약점 신고 방법

이 저장소에서 보안 취약점을 발견하셨다면, **공개 이슈로 등록하지 말고** 아래 방법 중 하나로 알려주세요.

1. **GitHub Security Advisory** (권장): 저장소의 [Security 탭 → Report a vulnerability](https://github.com/EvidenceChunker/Evidence-Chunker/security/advisories/new)에서 비공개로 신고할 수 있습니다.
2. **이메일**: 위 방법을 쓸 수 없다면 유지관리자에게 직접 이메일로 연락해 주세요.

### 신고 시 포함해주시면 좋은 정보

- 취약점 종류와 영향 범위
- 재현 방법(가능하다면 PoC 코드나 예제 PDF)
- 영향받는 버전

### 처리 절차

신고 접수 후 최대한 빠르게 확인 후 답변드리며, 취약점이 확인되면 패치 배포 후 신고자에게 크레딧을 남기는 것을 원칙으로 합니다(원치 않으실 경우 익명 처리 가능).
