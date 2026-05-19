import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ProfileDropdown } from '../ProfileDropdown/ProfileDropdown';
import { logoutUser } from '../../shared/api/authApi';
import { changeMyPassword, getMyFioRole } from '../../shared/api/userApi';
import { ChangePasswordModal } from '../../features/change-password/ChangePasswordModal';
import type { UserRole } from '../../entities/user/model/types';
import HeartlogoIcon from '../../shared/assets/icons/heartLogoIcon.svg';
import cls from './Header.module.scss';

export const Header = () => {
  const navigate = useNavigate();
  const [profile, setProfile] = useState<{ fio: string; role: UserRole }>({
    fio: 'Пользователь',
    role: 'user',
  });
  const [isChangePasswordOpen, setIsChangePasswordOpen] = useState(false);

  useEffect(() => {
    getMyFioRole().then(setProfile).catch(() => undefined);
  }, []);

  const handleLogout = () => {
    logoutUser();
    navigate('/login', { replace: true });
  };

  return (
    <>
      <header className={cls.header}>
        <div className={cls.logoBlock}>
          <img src={HeartlogoIcon} alt="CTReportUser" className={cls.logoIcon} />
          <span className={cls.logoText}>CTReport</span>
        </div>

        <ProfileDropdown
          triggerLabel={profile.fio}
          fullName={profile.fio}
          roleLabel={profile.role === 'admin' ? 'Администратор' : 'Пользователь'}
          showChangePassword
          onChangePassword={() => setIsChangePasswordOpen(true)}
          onLogout={handleLogout}
        />
      </header>

      <ChangePasswordModal
        isOpen={isChangePasswordOpen}
        onClose={() => setIsChangePasswordOpen(false)}
        onSubmit={changeMyPassword}
      />
    </>
  );
};
