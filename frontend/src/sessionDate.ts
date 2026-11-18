import type { ApplicationSession } from "./api";

export function sessionDate(session: ApplicationSession): string {
  const options: Intl.DateTimeFormatOptions = {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: session.timezone || "America/Los_Angeles",
  };
  const created = new Date(session.created_at);
  try {
    return new Intl.DateTimeFormat(undefined, options).format(created);
  } catch {
    return new Intl.DateTimeFormat(undefined, { ...options, timeZone: "America/Los_Angeles" }).format(created);
  }
}
