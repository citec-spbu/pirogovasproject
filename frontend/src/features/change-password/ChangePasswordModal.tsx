import { useState } from 'react';
import { Modal } from '../../shared/ui/Modal/Modal';
import { Button } from '../../shared/ui/Button/Button';
import { Input } from '../../shared/ui/Input/Input';
import CrossIcon from '../../shared/assets/icons/exitIcon.svg';
import cls from './ChangePasswordModal.module.scss';

interface ChangePasswordModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (oldPassword: string, newPassword: string) => Promise<void>;
}

export const ChangePasswordModal = ({
  isOpen,
  onClose,
  onSubmit,
}: ChangePasswordModalProps) => {
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [submitError, setSubmitError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!oldPassword.trim() || !newPassword.trim()) {
      setPasswordError('Введите старый и новый пароль');
      return;
    }

    setPasswordError('');
    setSubmitError('');
    setIsSubmitting(true);

    try {
      await onSubmit(oldPassword, newPassword);
      setOldPassword('');
      setNewPassword('');
      onClose();
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error.message : 'Не удалось сменить пароль'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className={cls.card}>
        <button type="button" className={cls.closeButton} onClick={onClose}>
          <img src={CrossIcon} alt="" className={cls.closeIcon} />
        </button>

        <h2 className={cls.title}>Смена пароля</h2>

        <div className={cls.formRow}>
          <p className={cls.label}>Старый пароль</p>

          <div className={cls.inputWrapper}>
            <Input
              label=""
              type="password"
              value={oldPassword}
              onChange={setOldPassword}
              variant="default"
              placeholder="old_password"
              error={passwordError}
            />
          </div>
        </div>

        <div className={cls.formRow}>
          <p className={cls.label}>Новый пароль</p>

          <div className={cls.inputWrapper}>
            <Input
              label=""
              type="password"
              value={newPassword}
              onChange={setNewPassword}
              variant="default"
              placeholder="new_password"
            />
          </div>
        </div>

        {submitError && <p className={cls.error}>{submitError}</p>}

        <Button
          type="button"
          className={cls.submitButton}
          onClick={handleSubmit}
        >
          {isSubmitting ? 'Сохраняется...' : 'Сохранить'}
        </Button>
      </div>
    </Modal>
  );
};
