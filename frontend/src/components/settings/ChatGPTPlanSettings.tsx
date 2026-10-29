import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type ChatGPTPlanStatus } from "@/lib/api";

export function ChatGPTPlanSettings({ onStatus }: { onStatus: (ready: boolean) => void }) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<ChatGPTPlanStatus | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const read = async () => {
      try {
        const next = await api.getChatGPTPlanStatus();
        if (!alive) return;
        setStatus(next);
        onStatus(next.plan_enabled);
        if (!next.pending) setUrl("");
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : String(err));
      }
    };
    void read();
    const timer = setInterval(() => void read(), 3000);
    return () => { alive = false; clearInterval(timer); onStatus(false); };
  }, [onStatus]);

  const connect = async (newAccount = false) => {
    setBusy(true);
    setError("");
    try {
      const result = await api.startChatGPTPlanLogin(newAccount);
      setUrl(result.authorization_url);
      setStatus(await api.getChatGPTPlanStatus());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setBusy(false); }
  };

  return <div className="space-y-3 border-t pt-4 text-sm" aria-label={t("settings.chatgptPlanTitle")}>
    <p className="font-medium">{t("settings.chatgptPlanTitle")}</p>
    <p className="text-xs text-muted-foreground">{t("settings.chatgptPlanHint")}</p>
    <p role="status">{status?.plan_enabled ? t("settings.chatgptPlanEnabled")
      : status?.connected ? t("settings.chatgptPlanDenied") : t("settings.chatgptPlanDisconnected")}</p>
    {!!status?.profiles.length && <label className="grid gap-1 text-xs">
      {t("settings.chatgptPlanConnection")}
      <select className="rounded-md border bg-background px-3 py-2 text-sm" value={status.active_profile ?? ""}
        disabled={busy || status.pending}
        onChange={async (event) => {
          try {
            const next = await api.selectChatGPTPlanProfile(event.target.value);
            setStatus(next); onStatus(next.plan_enabled);
          } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
        }}>
        {status.profiles.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
      </select>
    </label>}
    {status?.pending ? <div className="space-y-2">
      <p className="text-xs text-muted-foreground">{t("settings.chatgptPlanPending")}</p>
      {url && <a href={url} target="_blank" rel="noreferrer"
        className="inline-flex rounded-md border px-3 py-2 font-medium hover:bg-muted">Continue with ChatGPT</a>}
      <button type="button" className="ms-3 text-xs underline" onClick={async () => {
        try { setStatus(await api.cancelChatGPTPlanLogin()); setUrl(""); }
        catch (err) { setError(err instanceof Error ? err.message : String(err)); }
      }}>{t("settings.chatgptPlanCancel")}</button>
    </div> : <div className="flex flex-wrap items-center gap-3">
      <button type="button" onClick={() => void connect()} disabled={busy}
        className="rounded-md border px-3 py-2 font-medium hover:bg-muted disabled:opacity-50">Continue with ChatGPT</button>
      {status?.connected && <button type="button" className="text-xs underline" disabled={busy}
        onClick={() => void connect(true)}>{t("settings.chatgptPlanAddAccount")}</button>}
    </div>}
    <a className="inline-block text-xs underline" href="https://chatgpt.com/settings/usage" target="_blank" rel="noreferrer">
      {t("settings.chatgptPlanUsage")}
    </a>
    {(error || status?.error) && <p role="alert" className="break-words text-xs text-destructive">{error || status?.error}</p>}
  </div>;
}
