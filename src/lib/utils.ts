import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { format } from "date-fns";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function safeJsonParse<T = any>(json: string, fallback: T): T {
  try {
    return JSON.parse(json);
  } catch {
    return fallback;
  }
}

/**
 * Sanitize user-provided text before interpolating into AI prompts.
 * Truncates to maxLen and strips characters that could be used for prompt injection.
 */
export function sanitizeForPrompt(value: string | null | undefined, maxLen = 200): string {
  if (!value) return "";
  return value
    .slice(0, maxLen)
    .replace(/[\r\n]+/g, " ")
    .replace(/[{}[\]]/g, "")
    .trim();
}

export function formatDate(date: string | Date | null): string {
  if (!date) return "-";
  return format(new Date(date), "yyyy-MM-dd");
}

export function formatAmount(amount: number | null): string {
  if (amount === null || amount === undefined) return "-";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    minimumFractionDigits: 2,
  }).format(amount);
}

const DIGITS = ["零", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖"];
const UNITS = ["", "拾", "佰", "仟"];
const BIG_UNITS = ["", "万", "亿", "兆"];

function sectionToChinese(section: number): string {
  const digits = section.toString().split("").map(Number);
  let result = "";
  let zeroFlag = false;

  for (let i = 0; i < digits.length; i++) {
    const unitIndex = digits.length - 1 - i;
    if (digits[i] === 0) {
      zeroFlag = true;
    } else {
      if (zeroFlag) {
        result += "零";
        zeroFlag = false;
      }
      result += DIGITS[digits[i]] + UNITS[unitIndex];
    }
  }
  return result;
}

export function amountToChineseUppercase(amount: number): string {
  if (amount === 0) return "零元整";

  const isNegative = amount < 0;
  amount = Math.abs(amount);

  const [intPart, decPart] = amount.toFixed(2).split(".");
  const jiao = parseInt(decPart[0]);
  const fen = parseInt(decPart[1]);

  let intNum = parseInt(intPart);
  let result = isNegative ? "负" : "";

  if (intNum > 0) {
    const sections: number[] = [];
    while (intNum > 0) {
      sections.push(intNum % 10000);
      intNum = Math.floor(intNum / 10000);
    }

    for (let i = sections.length - 1; i >= 0; i--) {
      if (sections[i] === 0) continue;
      const sectionStr = sectionToChinese(sections[i]);
      // Add zero prefix if section is less than 1000 and not the highest section
      if (i < sections.length - 1 && sections[i] < 1000) {
        result += "零";
      }
      result += sectionStr + BIG_UNITS[i];
    }
    result += "元";
  }

  if (jiao === 0 && fen === 0) {
    result += "整";
  } else {
    if (jiao > 0) {
      result += DIGITS[jiao] + "角";
    } else if (parseInt(intPart) > 0) {
      result += "零";
    }
    if (fen > 0) {
      result += DIGITS[fen] + "分";
    }
  }

  return result;
}
