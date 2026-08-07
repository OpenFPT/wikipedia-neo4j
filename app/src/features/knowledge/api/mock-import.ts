const delay = (milliseconds: number) =>
  new Promise<void>((resolve) => globalThis.setTimeout(resolve, milliseconds));

export interface MockDocument {
  chunks: number;
  id: string;
  name: string;
  status: "ready";
}

export async function simulateDocumentImport(
  file: File,
  pause: (milliseconds: number) => Promise<void> = delay
): Promise<MockDocument> {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!(extension && ["pdf", "md", "txt"].includes(extension))) {
    throw new TypeError("Only PDF, Markdown, and text files are supported");
  }

  await pause(450);
  return {
    id: `mock:${file.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    name: file.name,
    chunks: Math.max(4, Math.ceil(file.size / 1800)),
    status: "ready",
  };
}
