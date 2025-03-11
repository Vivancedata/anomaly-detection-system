import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines multiple class names with tailwind merge support
 * Rule: Clsx utility - Functional programming pattern, Utilities pattern
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formats a date using Intl.DateTimeFormat
 * Rule: Utility functions - Declarative programming pattern
 */
export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

/**
 * Formats a number with thousand separators and optional decimal places
 * Rule: Utility functions - Declarative programming pattern
 */
export function formatNumber(
  value: number,
  options: { decimals?: number; prefix?: string; suffix?: string } = {}
): string {
  const { decimals = 0, prefix = "", suffix = "" } = options;
  
  const formatted = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
  
  return `${prefix}${formatted}${suffix}`;
}
