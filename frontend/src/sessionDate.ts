import type { ApplicationSession } from "./api";

export function sessionDate(session: ApplicationSession): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: session.timezone || "America/Los_Angeles",
  }).format(new Date(session.created_at));
}
