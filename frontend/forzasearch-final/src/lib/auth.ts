import { SignJWT, jwtVerify } from "jose";
import bcrypt from "bcryptjs";

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || "forzasearch-secret");

export interface User {
  id: string;
  email: string;
  username: string;
  firstName: string;
  lastName: string;
  role: "user" | "admin";
  createdAt: string;
}

interface StoredUser extends User {
  passwordHash: string;
}

// In-memory user store (swap for PostgreSQL/Supabase later)
const users: Map<string, StoredUser> = new Map();

// Seed admin user on first load
(async () => {
  if (!users.has("admin@forzasearch.com")) {
    const hash = await bcrypt.hash("admin123", 10);
    users.set("admin@forzasearch.com", {
      id: "admin-001",
      email: "admin@forzasearch.com",
      username: "admin",
      firstName: "Admin",
      lastName: "ForzaSearch",
      role: "admin",
      passwordHash: hash,
      createdAt: new Date().toISOString(),
    });
  }
})();

export async function registerUser(data: {
  email: string;
  username: string;
  firstName: string;
  lastName: string;
  password: string;
}): Promise<{ user: User; token: string } | { error: string }> {
  const email = data.email.toLowerCase();
  if (users.has(email)) {
    return { error: "Email already registered" };
  }

  for (const u of users.values()) {
    if (u.username.toLowerCase() === data.username.toLowerCase()) return { error: "Username already taken" };
  }

  const hash = await bcrypt.hash(data.password, 10);
  const user: StoredUser = {
    id: `user-${Date.now()}`,
    email,
    username: data.username,
    firstName: data.firstName,
    lastName: data.lastName,
    role: "user",
    passwordHash: hash,
    createdAt: new Date().toISOString(),
  };
  users.set(email, user);

  const token = await createToken(user);
  const { passwordHash: _, ...safeUser } = user;
  return { user: safeUser, token };
}

export async function loginUser(
  identifier: string,
  password: string
): Promise<{ user: User; token: string } | { error: string }> {
  const normalized = identifier.toLowerCase();
  const user = users.get(normalized) ?? Array.from(users.values()).find((u) => u.username.toLowerCase() === normalized);
  if (!user) return { error: "Invalid email/username or password" };

  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) return { error: "Invalid email/username or password" };

  const token = await createToken(user);
  const { passwordHash: _, ...safeUser } = user;
  return { user: safeUser, token };
}

export async function createToken(user: User | StoredUser): Promise<string> {
  return new SignJWT({ sub: user.id, email: user.email, role: user.role })
    .setProtectedHeader({ alg: "HS256" })
    .setExpirationTime("7d")
    .sign(JWT_SECRET);
}

export async function verifyToken(token: string): Promise<User | null> {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET);
    const email = payload.email as string;
    const stored = users.get(email);
    if (!stored) return null;
    const { passwordHash: _, ...safeUser } = stored;
    return safeUser;
  } catch {
    return null;
  }
}
