import { describe, expect, test } from "bun:test";
import { translate } from "./i18n";

describe("interface translations", () => {
  test("keeps English and Vietnamese interpolation in parity", () => {
    expect(
      translate("en", "deleteChatDescription", { title: "Research" })
    ).toBe("“Research” and its messages will be removed.");
    expect(
      translate("vi", "deleteChatDescription", { title: "Nghiên cứu" })
    ).toBe("“Nghiên cứu” và các tin nhắn sẽ bị xóa.");
    expect(translate("vi", "knowledgeBases")).toBe("Kho tri thức");
  });
});
