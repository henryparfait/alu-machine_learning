-- Trigger that resets valid_email to 0 only when the email actually changes
CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
SET NEW.valid_email = IF(NEW.email != OLD.email, 0, NEW.valid_email);
