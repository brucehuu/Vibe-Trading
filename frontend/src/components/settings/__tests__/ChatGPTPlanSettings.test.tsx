import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ChatGPTPlanSettings } from "../ChatGPTPlanSettings";

const mock = vi.hoisted(() => ({ getChatGPTPlanStatus: vi.fn(), startChatGPTPlanLogin: vi.fn(),
  cancelChatGPTPlanLogin: vi.fn(), selectChatGPTPlanProfile: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: mock }));
const disconnected = { connected: false, plan_enabled: false, active_profile: null,
  profiles: [], pending: false, error: null };

beforeEach(() => { vi.clearAllMocks(); mock.getChatGPTPlanStatus.mockResolvedValue(disconnected); });

it("does not treat identity sign-in alone as permission to use the plan", async () => {
  const onStatus = vi.fn();
  mock.getChatGPTPlanStatus.mockResolvedValue({ ...disconnected, connected: true });
  render(<ChatGPTPlanSettings onStatus={onStatus} />);
  await screen.findByText("Signed in · ChatGPT plan usage was not authorized");
  expect(onStatus).toHaveBeenCalledWith(false);
});

it("starts explicit authorization and offers cancellation without activating the provider", async () => {
  const onStatus = vi.fn();
  render(<ChatGPTPlanSettings onStatus={onStatus} />);
  await screen.findByText("ChatGPT is not connected");
  mock.startChatGPTPlanLogin.mockResolvedValue({ authorization_url: "https://auth.openai.com/test-authorization" });
  mock.getChatGPTPlanStatus.mockResolvedValue({ ...disconnected, pending: true });
  fireEvent.click(screen.getByRole("button", { name: "Continue with ChatGPT" }));
  expect(await screen.findByRole("link", { name: "Continue with ChatGPT" })).toHaveAttribute("href", "https://auth.openai.com/test-authorization");
  expect(mock.startChatGPTPlanLogin).toHaveBeenCalledWith(false);
  mock.cancelChatGPTPlanLogin.mockResolvedValue(disconnected);
  fireEvent.click(screen.getByRole("button", { name: "Cancel sign-in" }));
  await waitFor(() => expect(screen.queryByRole("link", { name: "Continue with ChatGPT" })).not.toBeInTheDocument());
  expect(onStatus).not.toHaveBeenCalledWith(true);
});

it("reports plan permission only from the backend's verified connection status", async () => {
  const onStatus = vi.fn();
  mock.getChatGPTPlanStatus.mockResolvedValue({ ...disconnected, connected: true, plan_enabled: true });
  render(<ChatGPTPlanSettings onStatus={onStatus} />);
  await screen.findByText("Connected · ChatGPT plan usage allowed");
  expect(onStatus).toHaveBeenCalledWith(true);
});
