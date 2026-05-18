import { Dropdown } from '../../shared/ui/Dropdown/Dropdown';
import userIcon from '../../shared/assets/icons/userIcon.svg';
import lockIcon from '../../shared/assets/icons/lockIcon.svg'
import cls from './ProfileDropdown.module.scss';

type ProfileDropdownProps = {
  triggerLabel: string;
  fullName: string;
  roleLabel: string;
  showChangePassword?: boolean;
  onChangePassword?: () => void;
  onLogout: () => void;
};

export const ProfileDropdown = ({
  triggerLabel,
  fullName,
  roleLabel,
  showChangePassword = false,
  onChangePassword,
  onLogout,
}: ProfileDropdownProps) => {
  return (
    <Dropdown
      trigger={
        <span className={cls.triggerContent}>
          <span>{triggerLabel}</span>
          <span className={cls.arrow} />
        </span>
      }
      triggerClassName={cls.trigger}
      menuClassName={cls.menu}
      align="right"
    >
      {(close) => (
        <>
          <div className={cls.userInfo}>
            <div className={cls.avatar}>
              <img src={userIcon} alt="" className={cls.avatarIcon} />
            </div>

            <div>
              <p className={cls.fullName}>{fullName}</p>
              <p className={cls.role}>{roleLabel}</p>
            </div>
          </div>

          <div className={cls.divider} />
          <div className={cls.containerButton}>
                {showChangePassword && (
                    <button type="button" className={cls.menuItem}
                    onClick={() => {
                        close();
                        onChangePassword?.();
                    }}
                    >
                    <img src={lockIcon} alt="" className={cls.lockIcon} />
                    <span>Сменить пароль</span>
                    </button>
                )}

                <button type="button" className={cls.logoutButton} 
                    onClick={() => {
                        close();
                        onLogout();
                        }} 
                    >
                    Выйти
                </button>
          </div>
        </>
      )}
    </Dropdown>
  );
};