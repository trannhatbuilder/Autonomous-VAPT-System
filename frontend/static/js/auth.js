/**
 * VAPT-AI Auth — W7-D-v2.
 *
 * Login page: email + password form.
 * On success → store tokens in localStorage → redirect to /chat.
 */

import { apiPost, navigate, toast, setTokens, clearTokens } from './utils.js';

export async function renderLogin(container) {
  container.innerHTML = `
    <div class="login-page">
      <div class="login-card">
        <div class="login-logo">
          <div class="login-logo-icon">🛡️</div>
          <div class="login-title">VAPT-AI</div>
          <div class="login-subtitle">Autonomous Pentest Platform</div>
        </div>
        <form class="login-form" id="login-form">
          <div>
            <label for="login-email">Email</label>
            <input type="email" id="login-email" class="input" placeholder="admin@vapt-ai.local" required autofocus />
          </div>
          <div>
            <label for="login-password">Password</label>
            <input type="password" id="login-password" class="input" placeholder="••••••••" required />
          </div>
          <div id="login-error" class="login-error hidden"></div>
          <button type="submit" class="btn btn-primary btn-lg">Sign in</button>
        </form>
      </div>
    </div>
  `;

  const form = document.getElementById('login-form');
  const errorDiv = document.getElementById('login-error');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    const submitBtn = form.querySelector('button[type="submit"]');

    submitBtn.disabled = true;
    submitBtn.textContent = 'Signing in...';
    errorDiv.classList.add('hidden');

    try {
      const result = await apiPost('/api/auth/login', { email, password });
      // Store tokens in localStorage (backend returns them in JSON body, not cookies)
      if (result.access_token) {
        setTokens(result.access_token, result.refresh_token);
        toast.success('Welcome back!');
        navigate('/chat');
      } else {
        throw new Error('No access token in response');
      }
    } catch (err) {
      errorDiv.textContent = err.message || 'Login failed';
      errorDiv.classList.remove('hidden');
      submitBtn.disabled = false;
      submitBtn.textContent = 'Sign in';
    }
  });
}