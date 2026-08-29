# llmwiki 작업 규칙 (Cline)

이 프로젝트는 `llmwiki` 컨텍스트 저장소를 사용합니다.

**작업을 시작하기 전에 아래 경로의 `AGENTS.md`를 먼저 읽고, 그 안의 규칙(`rules/`)과
스킬(`skills/`) 지침을 그대로 따르세요.**

```
/media/jeano/nvme-usb/llmwiki/AGENTS.md
```

요약:
1. 시작 시: `/media/jeano/nvme-usb/llmwiki/context/current/STATE.md` +
   `logs/activity.log.md` 최근 항목을 읽는다.
2. 종료/체크포인트 시: `STATE.md` 갱신 + `logs/activity.log.md`에 append.
3. `llmwiki`는 이 프로젝트(`8G_Qwen3-Coder-30B-A3B`)와 별개의 git 저장소이며, USB 드라이브의
   고정 절대 경로에 있다. 이 프로젝트의 git 이력에는 포함되지 않는다.
