from pathlib import Path

RAW_DIR = Path("data/raw")

DOCS = {
    "password_reset.md": """# Password Reset
## Summary
Use the self-service portal first. If it fails, contact Support.

## Steps
1. Go to the Password Portal.
2. Verify your identity (MFA).
3. Set a new password (min 12 chars, 1 number, 1 symbol).
4. Wait 5 minutes for sync.
5. If locked out, open a ticket with "PASSWORD LOCKOUT".

## Notes
- Do not share passwords in chat or email.
- Password resets may take up to 10 minutes to propagate.
""",
    "mfa_setup.md": """# MFA Setup
## Summary
MFA is required for VPN, email, and admin tools.

## Steps
1. Install Authenticator app on your phone.
2. Scan the QR code from the MFA enrollment page.
3. Save backup codes in a secure place.
4. Confirm enrollment with a 6-digit code.

## Troubleshooting
- If codes fail, check phone time sync.
- If you lost your phone, request MFA reset via ticket.
""",
    "vpn_troubleshooting.md": """# VPN Troubleshooting
## Common Issues
- Wrong credentials
- MFA prompt not appearing
- Network blocks

## Fixes
1. Confirm username format: firstname.lastname
2. Reset password if recently changed (wait 5-10 minutes).
3. Re-enroll MFA if prompts do not appear.
4. Try a different network (hotspot) if corporate Wi-Fi blocks VPN.
5. If error says "certificate missing", reinstall VPN client.

## Escalation
Include: device type, OS version, timestamp, screenshot of error.
""",
    "access_request.md": """# Access Request Process
## Summary
Access is granted by role and requires manager approval.

## Steps
1. Submit an access request form.
2. Include business justification and required role.
3. Manager approval is required.
4. Security reviews elevated roles.
5. Target SLA: 2 business days.

## Notes
- Do not request admin access unless required.
- Access changes are logged for audit.
""",
    "incident_comms.md": """# Incident Communication
## Severity Levels
- SEV1: Full outage / major business impact
- SEV2: Partial outage / degraded performance
- SEV3: Minor issue / workaround available

## Communication Rules
- SEV1: Updates every 30 minutes
- SEV2: Updates every 60 minutes
- SEV3: Updates every 24 hours or as needed

## Template
Status: Investigating / Identified / Monitoring / Resolved
Impact: Who is affected + what is broken
Next Update: Time
""",
    "ticket_triage.md": """# Ticket Triage Guide
## Required Fields
- Issue summary
- Steps to reproduce
- Expected vs actual
- Logs/screenshots
- Urgency + business impact

## Routing Rules
- Password/MFA -> Identity team
- VPN -> Network team
- Access -> IAM queue
- Outage -> Incident channel (follow incident comms)

## Good vs Bad Tickets
Good: clear steps + evidence.
Bad: vague, no repro, missing impact.
""",
}

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in DOCS.items():
        (RAW_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"✅ Wrote {len(DOCS)} dummy docs to {RAW_DIR.resolve()}")

if __name__ == "__main__":
    main()
